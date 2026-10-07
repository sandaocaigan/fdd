#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-/home/user/lichengyuan/.conda_envs/feddistill310/bin/python}"
MANIFEST="splits/cifar10_public_500x10_seed20260805.json"

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
    pgrep -f '^/home/user/lichengyuan/.conda_envs/feddistill310/bin/python -u main.py' || true
)
if [ "${#ACTIVE_TRAINING[@]}" -gt 0 ]; then
    echo "Existing training process(es) detected: ${ACTIVE_TRAINING[*]}"
    echo "Stop or finish them before starting this validation group."
    exit 1
fi

echo "Running fixed-public split tests."
"$PYTHON" -m unittest tests.test_data_splits -v || exit $?

run_exp() {
    local run_id="$1"
    local name="$2"
    local attack="$3"
    local n_byzantine="$4"
    local use_spc="$5"
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
        echo "Attack: $attack"
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
        --attack "$attack" \
        --defence NoDefence \
        --memory_method none \
        --n_clients 30 \
        --n_byzantine_clients "$n_byzantine" \
        --n_total_local_epochs 70 \
        --n_communications 70 \
        --n_server_epochs_per_round 1 \
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
        "${probe_args[@]}" \
        >> "$log" 2>&1

    local status=$?
    {
        echo "Exit code: $status"
        echo "End: $(date)"
    } >> "$log"

    echo "Run $run_id finished with exit code $status"
    grep "Final accuracy summary" "$log" | tail -n 1 || true
    return "$status"
}

echo "Fixed-public validation group started: $(date)"

run_exp 10401 clear_mean_fixed_public NoAttack 0 false || exit $?
run_exp 10402 clear_spc_mean_fixed_public NoAttack 0 true || exit $?
run_exp 10403 shuffling_mean_fixed_public Shuffling 9 false || exit $?
run_exp 10404 shuffling_spc_mean_fixed_public Shuffling 9 true || exit $?

echo "All fixed-public validation experiments finished: $(date)"
