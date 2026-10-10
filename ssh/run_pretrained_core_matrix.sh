#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-/home/user/lichengyuan/.conda_envs/feddistill310/bin/python}"
MANIFEST="splits/cifar10_public_500x10_seed20260805.json"
CHECKPOINT="checkpoints/cifar10_resnet18_public5000_seed20260805.pt"
PRETRAIN_LOG="logs/10500_public5000_pretrain.log"
SUMMARY="logs/10501_10504_pretrained_core_matrix.summary.txt"

cd "$PROJECT_DIR" || {
    echo "Cannot enter project directory: $PROJECT_DIR"
    exit 1
}

mkdir -p logs checkpoints
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ ! -x "$PYTHON" ]; then
    echo "Python executable not found: $PYTHON"
    exit 1
fi
if [ ! -f "$MANIFEST" ]; then
    echo "Fixed public split manifest not found: $MANIFEST"
    exit 1
fi
if pgrep -af '[p]ython.*main.py' >/dev/null; then
    echo "Another main.py training process is active; refusing to start."
    exit 1
fi
if pgrep -af '[p]ython.*tools/train_clean_teacher.py' >/dev/null; then
    echo "Another server pretraining process is active; refusing to start."
    exit 1
fi

for run_log in \
    logs/10501_clear_pretrained_no_spc.log \
    logs/10502_manipkd_pl_pretrained_no_spc.log \
    logs/10503_clear_pretrained_spc.log \
    logs/10504_manipkd_pl_pretrained_spc.log; do
    if [ -e "$run_log" ]; then
        echo "Experiment log already exists: $run_log"
        echo "Refusing to overwrite it."
        exit 1
    fi
done
if [ -e "$SUMMARY" ]; then
    echo "Summary already exists: $SUMMARY"
    echo "Refusing to mix a previous launch with this matrix."
    exit 1
fi

echo "Running preflight tests."
"$PYTHON" -m unittest \
    tests.test_server_pretraining \
    tests.test_teacher_public_split \
    tests.test_data_splits \
    tests.test_manipulatingkd_history \
    -v || exit $?

if [ -f "$CHECKPOINT" ]; then
    echo "Using existing server checkpoint: $CHECKPOINT"
else
    if [ -e "$PRETRAIN_LOG" ]; then
        echo "Pretraining log exists but checkpoint is missing: $PRETRAIN_LOG"
        echo "Move the old log aside before retrying so evidence is not overwritten."
        exit 1
    fi

    {
        echo "=================================================="
        echo "Start: $(date)"
        echo "Task: supervised server initialization"
        echo "Training data: fixed labeled CIFAR-10 public split (5000 samples)"
        echo "Validation data: fixed CIFAR-10 validation split (2500 samples)"
        echo "Private client data used: no"
        echo "Python: $PYTHON"
        echo "=================================================="
    } > "$PRETRAIN_LOG"

    "$PYTHON" -u tools/train_clean_teacher.py \
        --dataset cifar10 \
        --arch ResNet18 \
        --output "$CHECKPOINT" \
        --public_split_manifest "$MANIFEST" \
        --public_samples_per_class 500 \
        --public_val_samples_per_class 250 \
        --public_split_seed 20260805 \
        --seed 20260805 \
        --epochs 100 \
        --batch_size 128 \
        --lr 0.1 \
        --device cuda:0 \
        --use_amp \
        >> "$PRETRAIN_LOG" 2>&1

    pretrain_status=$?
    {
        echo "Exit code: $pretrain_status"
        echo "End: $(date)"
    } >> "$PRETRAIN_LOG"
    if [ "$pretrain_status" -ne 0 ]; then
        echo "Server pretraining failed. See $PRETRAIN_LOG"
        exit "$pretrain_status"
    fi
fi

run_exp() {
    local run_id="$1"
    local name="$2"
    local attack="$3"
    local n_byzantine="$4"
    local use_spc="$5"
    local log="logs/${run_id}_${name}.log"
    local attack_args=()
    local probe_args=()

    if [ "$attack" = "ManipulatingKD" ]; then
        attack_args=(
            --manipkd_formulation paper
            --manipkd_knowledge partial_local
            --manipkd_history_rounds 1
            --manipkd_noise_std 0.10
            --manipkd_lr 0.50
            --manipkd_steps 60
            --manipkd_lagrange_lr 1.0
            --manipkd_project true
        )
    fi

    if [ "$use_spc" = "true" ]; then
        probe_args=(
            --use_probe true
            --probe_mode replace
            --probe_type scale
            --probe_score hybrid
            --probe_base_count 20
            --probe_scales 1.0,1.2,1.5,2.0
            --probe_seed 0
            --probe_start_round 15
            --probe_core_sigma_audit true
            --probe_core_sigma_filter true
            --probe_core_sigma_threshold 5.0
        )
    else
        probe_args=(
            --use_probe false
            --probe_core_sigma_audit false
            --probe_core_sigma_filter false
        )
    fi

    {
        echo "=================================================="
        echo "Start: $(date)"
        echo "Run ID: $run_id"
        echo "Attack: $attack"
        echo "Defence: NoDefence (arithmetic mean)"
        echo "SPC: $use_spc"
        echo "Server initialization: $CHECKPOINT"
        echo "Public split: fixed CIFAR-10, 500 samples per class"
        echo "Python: $PYTHON"
        echo "=================================================="
    } > "$log"

    "$PYTHON" -u main.py \
        --debug \
        --run_id "$run_id" \
        --seed 20260805 \
        --dataset cifar10 \
        --arch ResNet18 \
        --batch_size 128 \
        --strategy FedDistill \
        --attack "$attack" \
        --defence NoDefence \
        --memory_method none \
        --n_clients 30 \
        --n_byzantine_clients "$n_byzantine" \
        --n_total_local_epochs 70 \
        --n_communications 70 \
        --n_server_epochs_per_round 5 \
        --server_lr "(Linear, 0.01)" \
        --client_lr "(Linear, 0.1)" \
        --restart_client_lr false \
        --reinit_server false \
        --server_init_ckpt "$CHECKPOINT" \
        --distill_temperature 1.0 \
        --attack_start_round 15 \
        --use_amp false \
        --use_diffusion_purifier false \
        --public_ds none \
        --public_ds_fraction none \
        --public_split_mode fixed_stratified \
        --public_samples_per_class 500 \
        --public_val_samples_per_class 250 \
        --public_split_seed 20260805 \
        --public_split_manifest "$MANIFEST" \
        "${attack_args[@]}" \
        "${probe_args[@]}" \
        >> "$log" 2>&1

    local status=$?
    {
        echo "Exit code: $status"
        echo "End: $(date)"
    } >> "$log"

    {
        echo "Run $run_id: exit=$status, attack=$attack, SPC=$use_spc"
        grep -F "Server initialization: loaded public-only checkpoint" "$log" | tail -n 1 || true
        grep "Final accuracy summary" "$log" | tail -n 1 || true
        grep -F "[ManipulatingKD diagnostics]" "$log" | tail -n 1 || true
        grep -F "[Probe Core-Sigma Filter]" "$log" | tail -n 1 || true
        echo
    } | tee -a "$SUMMARY"
    return "$status"
}

{
    echo "Pretrained core matrix started: $(date)"
    echo "Checkpoint: $CHECKPOINT"
    echo "Public dataset: fixed labeled CIFAR-10, 5000 samples"
    echo "Shared seed: 20260805"
    echo "Runs: 10501 clear/no-SPC; 10502 P-L/no-SPC; 10503 clear/SPC; 10504 P-L/SPC"
    echo
} > "$SUMMARY"

run_exp 10501 clear_pretrained_no_spc NoAttack 0 false || exit $?
run_exp 10502 manipkd_pl_pretrained_no_spc ManipulatingKD 9 false || exit $?
run_exp 10503 clear_pretrained_spc NoAttack 0 true || exit $?
run_exp 10504 manipkd_pl_pretrained_spc ManipulatingKD 9 true || exit $?

echo "All pretrained core experiments finished: $(date)" | tee -a "$SUMMARY"
