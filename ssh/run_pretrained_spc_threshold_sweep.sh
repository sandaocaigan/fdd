#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-/home/user/lichengyuan/.conda_envs/feddistill310/bin/python}"
MANIFEST="splits/cifar10_public_500x10_seed20260805.json"
CHECKPOINT="checkpoints/cifar10_resnet18_public5000_seed20260805.pt"
SUMMARY="logs/10505_10512_pretrained_spc_threshold.summary.txt"

cd "$PROJECT_DIR" || {
    echo "Cannot enter project directory: $PROJECT_DIR"
    exit 1
}

mkdir -p logs
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
if [ ! -f "$CHECKPOINT" ]; then
    echo "Server pretraining checkpoint not found: $CHECKPOINT"
    exit 1
fi
if pgrep -af '[p]ython.*main.py|[p]ython.*train_clean_teacher.py' >/dev/null; then
    echo "Another training or pretraining process is active; refusing to start."
    exit 1
fi

for run_log in \
    logs/10505_clear_spc_sigma5.log \
    logs/10506_manipkd_spc_sigma5.log \
    logs/10507_clear_spc_sigma6.log \
    logs/10508_manipkd_spc_sigma6.log \
    logs/10509_clear_spc_sigma7.log \
    logs/10510_manipkd_spc_sigma7.log \
    logs/10511_clear_spc_sigma8.log \
    logs/10512_manipkd_spc_sigma8.log; do
    if [ -e "$run_log" ]; then
        echo "Experiment log already exists: $run_log"
        echo "Refusing to overwrite it."
        exit 1
    fi
done
if [ -e "$SUMMARY" ]; then
    echo "Summary already exists: $SUMMARY"
    echo "Refusing to mix a previous sweep with this run."
    exit 1
fi

echo "Running preflight tests."
"$PYTHON" -m unittest \
    tests.test_server_pretraining \
    tests.test_teacher_public_split \
    tests.test_data_splits \
    tests.test_manipulatingkd_history \
    -v || exit $?

run_exp() {
    local run_id="$1"
    local name="$2"
    local attack="$3"
    local n_byzantine="$4"
    local threshold="$5"
    local log="logs/${run_id}_${name}.log"
    local attack_args=()

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

    {
        echo "=================================================="
        echo "Start: $(date)"
        echo "Run ID: $run_id"
        echo "Attack: $attack"
        echo "Defence: NoDefence (arithmetic mean)"
        echo "SPC: true"
        echo "Probe core-sigma threshold: $threshold"
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
        --use_probe true \
        --probe_mode replace \
        --probe_type scale \
        --probe_score hybrid \
        --probe_base_count 20 \
        --probe_scales 1.0,1.2,1.5,2.0 \
        --probe_seed 0 \
        --probe_start_round 15 \
        --probe_core_sigma_audit true \
        --probe_core_sigma_filter true \
        --probe_core_sigma_threshold "$threshold" \
        "${attack_args[@]}" \
        >> "$log" 2>&1

    local status=$?
    {
        echo "Exit code: $status"
        echo "End: $(date)"
    } >> "$log"

    {
        echo "Run $run_id: exit=$status, attack=$attack, threshold=$threshold"
        grep "Final accuracy summary" "$log" | tail -n 1 || true
        grep -F "[Probe Core-Sigma Filter]" "$log" | tail -n 1 || true
        grep -F "[ManipulatingKD diagnostics]" "$log" | tail -n 1 || true
        echo
    } | tee -a "$SUMMARY"
    return "$status"
}

{
    echo "Pretrained SPC threshold sweep started: $(date)"
    echo "Checkpoint: $CHECKPOINT"
    echo "Public dataset: fixed labeled CIFAR-10, 5000 samples"
    echo "Shared seed: 20260805"
    echo "Thresholds: 5.0, 6.0, 7.0, 8.0"
    echo "Runs: clean and P-L at every threshold"
    echo
} > "$SUMMARY"

run_exp 10505 clear_spc_sigma5 NoAttack 0 5.0 || exit $?
run_exp 10506 manipkd_spc_sigma5 ManipulatingKD 9 5.0 || exit $?
run_exp 10507 clear_spc_sigma6 NoAttack 0 6.0 || exit $?
run_exp 10508 manipkd_spc_sigma6 ManipulatingKD 9 6.0 || exit $?
run_exp 10509 clear_spc_sigma7 NoAttack 0 7.0 || exit $?
run_exp 10510 manipkd_spc_sigma7 ManipulatingKD 9 7.0 || exit $?
run_exp 10511 clear_spc_sigma8 NoAttack 0 8.0 || exit $?
run_exp 10512 manipkd_spc_sigma8 ManipulatingKD 9 8.0 || exit $?

echo "All pretrained SPC threshold experiments finished: $(date)" | tee -a "$SUMMARY"
