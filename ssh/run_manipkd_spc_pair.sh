#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-/home/user/lichengyuan/.conda_envs/feddistill310/bin/python}"
MANIFEST="splits/cifar10_public_500x10_seed20260805.json"
SUMMARY="logs/10425_manipkd_spc.summary.txt"

cd "$PROJECT_DIR" || {
    echo "Cannot enter project directory: $PROJECT_DIR"
    exit 1
}

mkdir -p logs
export WANDB_MODE=offline

if [ ! -x "$PYTHON" ]; then
    echo "Python executable not found: $PYTHON"
    exit 1
fi
if [ ! -f "$MANIFEST" ]; then
    echo "Fixed public split manifest not found: $MANIFEST"
    exit 1
fi

mapfile -t ACTIVE_TRAINING < <(
    pgrep -f "$PYTHON -u main.py" || true
)
if [ "${#ACTIVE_TRAINING[@]}" -gt 0 ]; then
    echo "Existing training process(es) detected: ${ACTIVE_TRAINING[*]}"
    echo "Stop or finish them before starting this experiment pair."
    exit 1
fi

RUN_ID=10425
RUN_LOG="logs/10425_manipkd_pl_fixed5000_spc.log"
if pgrep -af "[m]ain.py.*--run_id ${RUN_ID}([[:space:]]|$)" >/dev/null; then
    echo "Run ID $RUN_ID is already active; refusing a duplicate launch."
    exit 1
fi
if [ -e "$RUN_LOG" ]; then
    echo "Experiment log already exists: $RUN_LOG"
    echo "Refusing to overwrite it."
    exit 1
fi

echo "Running preflight tests."
"$PYTHON" -m unittest \
    tests.test_data_splits \
    tests.test_manipulatingkd_history \
    -v || exit $?

run_exp() {
    local run_id="$1"
    local name="$2"
    local use_spc="$3"
    local log="logs/${run_id}_${name}.log"
    local probe_args=()

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
        echo "Attack: ManipulatingKD P-L"
        echo "Defence: NoDefence (arithmetic mean)"
        echo "SPC: $use_spc"
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
        --attack ManipulatingKD \
        --defence NoDefence \
        --memory_method none \
        --n_clients 30 \
        --n_byzantine_clients 9 \
        --n_total_local_epochs 70 \
        --n_communications 70 \
        --n_server_epochs_per_round 5 \
        --server_lr "(Linear, 0.01)" \
        --client_lr "(Linear, 0.1)" \
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
        --manipkd_formulation paper \
        --manipkd_knowledge partial_local \
        --manipkd_history_rounds 1 \
        --manipkd_noise_std 0.10 \
        --manipkd_lr 0.50 \
        --manipkd_steps 60 \
        --manipkd_lagrange_lr 1.0 \
        --manipkd_project true \
        "${probe_args[@]}" \
        >> "$log" 2>&1

    local status=$?
    {
        echo "Exit code: $status"
        echo "End: $(date)"
    } >> "$log"

    {
        echo "Run $run_id: exit=$status, SPC=$use_spc"
        grep "Final accuracy summary" "$log" | tail -n 1 || true
        grep -F "[ManipulatingKD diagnostics]" "$log" | tail -n 1 || true
        grep -F "[ManipulatingKD P-L history]" "$log" | tail -n 1 || true
    } | tee -a "$SUMMARY"
    return "$status"
}

{
    echo "ManipulatingKD P-L/SPC retry started: $(date)"
    echo "Public dataset: fixed CIFAR-10, 5000 samples"
    echo "Shared seed: 20260805"
} > "$SUMMARY"

run_exp 10425 manipkd_pl_fixed5000_spc true || exit $?

echo "ManipulatingKD P-L/SPC retry finished: $(date)" | tee -a "$SUMMARY"
