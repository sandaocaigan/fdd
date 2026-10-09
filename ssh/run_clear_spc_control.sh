#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-/home/user/lichengyuan/.conda_envs/feddistill310/bin/python}"
MANIFEST="splits/cifar10_public_500x10_seed20260805.json"
RUN_ID=10426
LOG="logs/10426_clear_fixed5000_spc.log"
SUMMARY="logs/10426_clear_spc.summary.txt"

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
if pgrep -af "$PYTHON -u main.py" >/dev/null; then
    echo "Another training process is already active; refusing to start."
    exit 1
fi
if [ -e "$LOG" ]; then
    echo "Experiment log already exists: $LOG"
    echo "Refusing to overwrite it."
    exit 1
fi

echo "Running fixed-public split tests."
"$PYTHON" -m unittest tests.test_data_splits -v || exit $?

{
    echo "=================================================="
    echo "Start: $(date)"
    echo "Run ID: $RUN_ID"
    echo "Attack: NoAttack"
    echo "Defence: NoDefence (arithmetic mean)"
    echo "SPC: true"
    echo "Public split: fixed CIFAR-10, 500 samples per class"
    echo "Python: $PYTHON"
    echo "=================================================="
} > "$LOG"

"$PYTHON" -u main.py \
    --debug \
    --run_id "$RUN_ID" \
    --seed 20260805 \
    --dataset cifar10 \
    --arch ResNet18 \
    --batch_size 128 \
    --strategy FedDistill \
    --attack NoAttack \
    --defence NoDefence \
    --memory_method none \
    --n_clients 30 \
    --n_byzantine_clients 0 \
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
    --probe_core_sigma_threshold 5.0 \
    >> "$LOG" 2>&1

STATUS=$?
{
    echo "Exit code: $STATUS"
    echo "End: $(date)"
} >> "$LOG"

{
    echo "Run $RUN_ID: exit=$STATUS, attack=NoAttack, SPC=true"
    grep "Final accuracy summary" "$LOG" | tail -n 1 || true
    grep -F "[Probe Core-Sigma Filter]" "$LOG" | tail -n 1 || true
} | tee "$SUMMARY"

exit "$STATUS"
