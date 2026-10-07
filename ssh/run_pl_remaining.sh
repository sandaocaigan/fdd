#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-/home/user/lichengyuan/.conda_envs/feddistill310/bin/python}"

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

run_exp() {
    local run_id="$1"
    local name="$2"
    local noise="$3"
    local learning_rate="$4"
    local steps="$5"
    local log="logs/${run_id}_${name}.log"

    {
        echo "=================================================="
        echo "Start: $(date)"
        echo "Run ID: $run_id"
        echo "Noise: $noise"
        echo "Learning rate: $learning_rate"
        echo "Steps: $steps"
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
        --n_server_epochs_per_round 1 \
        --distill_temperature 1.0 \
        --attack_start_round 15 \
        --use_amp false \
        --use_probe false \
        --probe_core_sigma_audit false \
        --probe_core_sigma_filter false \
        --use_diffusion_purifier false \
        --public_ds_fraction none \
        --manipkd_formulation paper \
        --manipkd_knowledge partial_local \
        --manipkd_history_rounds 1 \
        --manipkd_noise_std "$noise" \
        --manipkd_lr "$learning_rate" \
        --manipkd_steps "$steps" \
        --manipkd_lagrange_lr 1.0 \
        >> "$log" 2>&1

    local status=$?
    {
        echo "Exit code: $status"
        echo "End: $(date)"
    } >> "$log"

    echo "Run $run_id finished with exit code $status"
    grep "Final accuracy summary" "$log" | tail -n 1 || true
    grep -F "[ManipulatingKD diagnostics]" "$log" | tail -n 1 || true
}

echo "Remaining P-L experiments started: $(date)"
run_exp 10302 manipkd_pl_fixed_default 0.01 0.05 30
run_exp 10303 manipkd_pl_moderate 0.10 0.50 60
run_exp 10304 manipkd_pl_strong 0.50 0.50 100
echo "All remaining P-L experiments finished: $(date)"
