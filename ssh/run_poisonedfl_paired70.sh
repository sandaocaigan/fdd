#!/usr/bin/env bash
set -Eeuo pipefail

# Eight paired branches from one round-14 state:
# strength {0,1,5,10} x SPC {off,on}, rounds 15--70.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-/home/user/lichengyuan/.conda_envs/feddistill310/bin/python}"
ARCH="${ARCH:-ResNet18}"
BASE_ID="${1:-11121}"
PROFILE="${2:-screen}"
MODE="${3:-plan}"
if [[ ! "$BASE_ID" =~ ^[1-9][0-9]{0,8}$ ]] ||
   [[ "$PROFILE" != screen && "$PROFILE" != smoke ]] ||
   [[ "$MODE" != plan && "$MODE" != run ]]; then
    echo "Usage: bash $0 [base_id] [screen|smoke] [plan|run]" >&2
    exit 2
fi

CLIENTS=30; BYZANTINE=9; ATTACK_START=15; BRANCH_ROUNDS=56; SAMPLES=2048
if [[ "$PROFILE" == smoke ]]; then
    CLIENTS=4; BYZANTINE=1; ATTACK_START=2; BRANCH_ROUNDS=2; SAMPLES=128
fi
LAST_ID=$((BASE_ID + 8))
WARMUP=$((ATTACK_START - 1))
CHECKPOINT="${PAIR_CHECKPOINT:-checkpoints/poisonedfl_pair_${BASE_ID}_round${WARMUP}.pt}"
SUMMARY="logs/${BASE_ID}_${LAST_ID}_poisonedfl_paired.summary.txt"

print_plan() {
    echo "PoisonedFL paired70: CIFAR-10/$ARCH"
    echo "Warmup=$WARMUP; branches=$BRANCH_ROUNDS rounds (15--70 in screen profile)"
    echo "Matrix: strength={0,1,5,10} x SPC={off,on}; clients=$CLIENTS; malicious=$BYZANTINE"
    echo "Checkpoint=$CHECKPOINT; LR horizon=70; fixed output-space directions; adaptive=false"
    for strength in 0 1 5 10; do
        for spc in nospc spc; do echo "condition=strength${strength}_${spc}"; done
    done
}
print_plan
[[ "$MODE" == plan ]] && exit 0
cd "$PROJECT_DIR"
mkdir -p logs checkpoints
[[ -x "$PYTHON" ]] || { echo "Python not found: $PYTHON" >&2; exit 1; }
command -v flock >/dev/null || { echo "flock is required" >&2; exit 1; }
exec 9>logs/.pcfdla_matrix.lock
flock -n 9 || { echo "Another experiment scheduler is active" >&2; exit 1; }
[[ ! -e "$SUMMARY" ]] || { echo "Existing summary: $SUMMARY; use a fresh base ID" >&2; exit 1; }
shopt -s nullglob
for ((id=BASE_ID; id<=LAST_ID; id++)); do
    for existing in logs/"${id}"_*; do
        [[ "$existing" == *.launcher.log || "$existing" == *.launcher.pid ]] && continue
        echo "Run ID already has output: $existing" >&2; exit 1
    done
done
export WANDB_MODE=offline
"$PYTHON" -m unittest discover -s tests -v
"$PYTHON" main.py --help | grep -- '--poisonedfl_pair_spc' >/dev/null
echo "Starting paired70 at $(date -Is)"
sha256sum main.py runners/PoisonedFLPairRunner.py runners/ScratchRunner.py \
    byzantine/attacks.py byzantine/poisonedfl.py ssh/run_poisonedfl_paired70.sh
"$PYTHON" -u main.py --debug --run_id "$BASE_ID" --seed 20260805 \
    --dataset cifar10 --arch "$ARCH" --batch_size 128 \
    --strategy FedDistill --attack PoisonedFL --defence NoDefence --memory_method none \
    --n_clients "$CLIENTS" --n_byzantine_clients "$BYZANTINE" \
    --n_total_local_epochs 70 --n_communications 70 --n_server_epochs_per_round 1 \
    --optimizer SGD --momentum 0.9 --weight_decay 0.0001 \
    --server_lr "(Linear, 0.1)" --client_lr "(Linear, 0.1)" \
    --restart_client_lr false --reinit_server false --warm_restarts false \
    --server_init_ckpt none --client_early_stopping false --server_early_stopping false \
    --distill_temperature 1 --attack_start_round "$ATTACK_START" \
    --poisonedfl_strength 1 --poisonedfl_direction fixed --poisonedfl_adaptive false \
    --poisonedfl_min_strength 0.01 --poisonedfl_max_strength 10 \
    --poisonedfl_pair_rounds "$BRANCH_ROUNDS" --poisonedfl_pair_samples "$SAMPLES" \
    --poisonedfl_pair_checkpoint "$CHECKPOINT" --poisonedfl_pair_spc true \
    --use_amp false --use_diffusion_purifier false --evaluate_backdoor_asr false \
    --public_split_mode legacy --public_ds stl10 --public_ds_fraction none \
    --use_probe true --probe_mode replace --probe_type scale --probe_score hybrid \
    --probe_base_count 20 --probe_scales 1.0,1.2,1.5,2.0 --probe_seed 0 \
    --probe_start_round "$ATTACK_START" --probe_core_sigma_audit true \
    --probe_core_sigma_filter false --probe_core_sigma_threshold 5
echo "Launcher exit=0; finished at $(date -Is)"
