#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-/home/user/lichengyuan/.conda_envs/feddistill310/bin/python}"
BASE_ID="${1:-11111}"
PROFILE="${2:-screen}"
MODE="${3:-run}"
if [[ ! "$BASE_ID" =~ ^[1-9][0-9]{0,8}$ ]] ||
   [[ "$PROFILE" != screen && "$PROFILE" != smoke ]] ||
   [[ "$MODE" != run && "$MODE" != plan ]]; then
    echo "Usage: bash $0 [base_id] [screen|smoke] [run|plan]"
    exit 2
fi
CLIENTS=30
BYZANTINE=9
ATTACK_START=15
# Complete rounds 15--70 after the shared clean rounds 1--14.
BRANCH_ROUNDS=56
SAMPLES=2048
if [[ "$PROFILE" == smoke ]]; then
    CLIENTS=4
    BYZANTINE=1
    ATTACK_START=2
    BRANCH_ROUNDS=2
    SAMPLES=128
fi
LAST_ID=$((BASE_ID + 4))
WARMUP=$((ATTACK_START - 1))
CHECKPOINT="${PAIR_CHECKPOINT:-checkpoints/poisonedfl_pair_${BASE_ID}_round${WARMUP}.pt}"
SUMMARY="logs/${BASE_ID}_${LAST_ID}_poisonedfl_paired.summary.txt"
NAMES=(warmup clear strength1 strength5 strength10)
print_plan() {
    echo "Paired output-space PoisonedFL adaptation; profile=$PROFILE; project=$PROJECT_DIR"
    echo "Clean warmup=$WARMUP rounds; each of 4 branches=$BRANCH_ROUNDS rounds"
    echo "Clients=$CLIENTS; malicious=$BYZANTINE in attacks; LR schedule horizon=70 rounds"
    echo "Mean; fixed directions; adaptive=false; SPC filter=false; probe audit=true"
    echo "CIFAR-10 private=47500; STL-10 public=100000; checkpoint=$CHECKPOINT"
    echo "Restore models, optimizer momentum, scheduler, scaler and RNG for every branch."
    echo "Warmup accuracy is recorded; the shared state is not assumed converged."
    for i in "${!NAMES[@]}"; do
        echo "$((BASE_ID + i)) ${NAMES[$i]}"
    done
}
if [[ "$MODE" == plan ]]; then print_plan; exit 0; fi
cd "$PROJECT_DIR"
mkdir -p logs
[[ -x "$PYTHON" ]] || { echo "Python not found: $PYTHON"; exit 1; }
command -v flock >/dev/null || { echo "flock is required."; exit 1; }
exec 9>logs/.pcfdla_matrix.lock
flock -n 9 || { echo "Another attack scheduler is active."; exit 1; }
if pgrep -af '[p]ython[^ ]* .*main\.py|[p]ython[^ ]* .*train_clean_teacher\.py'; then
    echo "Another training process is active; refusing to start."
    exit 1
fi
[[ ! -e "$SUMMARY" ]] || { echo "Existing summary: $SUMMARY; use fresh IDs."; exit 1; }
shopt -s nullglob
for ((id=BASE_ID; id<=LAST_ID; id++)); do
    for existing in logs/"${id}"_*; do
        [[ "$existing" == *.launcher.log || "$existing" == *.launcher.pid ]] && continue
        echo "Run ID already has output: $existing; choose fresh IDs."
        exit 1
    done
done
export WANDB_MODE=offline
print_plan
"$PYTHON" -m unittest discover -s tests -v
"$PYTHON" main.py --help | grep -- '--poisonedfl_pair_rounds' >/dev/null
"$PYTHON" - <<'PY'
from pathlib import Path
from torchvision.datasets import CIFAR10, STL10
from torchvision.datasets.utils import check_integrity
for train in (True, False):
    data = CIFAR10("datasets_pytorch/cifar10", train=train, download=False)
    assert len(data) == (50000 if train else 10000)
for name, md5 in STL10.train_list + STL10.test_list:
    path = Path("datasets_pytorch/stl10") / STL10.base_folder / name
    if not check_integrity(str(path), md5):
        raise RuntimeError(f"Missing or damaged STL-10 file: {path}")
print("CIFAR-10/STL-10 checks passed without downloads.")
PY
echo "Paired experiment started: $(date -Is)"
sha256sum main.py runners/PoisonedFLPairRunner.py runners/ScratchRunner.py \
    byzantine/attacks.py byzantine/poisonedfl.py ssh/run_poisonedfl_paired.sh
if "$PYTHON" -u main.py \
    --debug --run_id "$BASE_ID" --seed 20260805 \
    --dataset cifar10 --arch ResNet18 --batch_size 128 \
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
    --poisonedfl_pair_checkpoint "$CHECKPOINT" --poisonedfl_pair_spc false \
    --use_amp false --use_diffusion_purifier false --evaluate_backdoor_asr false \
    --public_split_mode legacy --public_ds stl10 --public_ds_fraction none \
    --use_probe true --probe_mode replace --probe_type scale --probe_score hybrid \
    --probe_base_count 20 --probe_scales 1.0,1.2,1.5,2.0 --probe_seed 0 \
    --probe_start_round "$ATTACK_START" --probe_core_sigma_audit true \
    --probe_core_sigma_filter false --probe_core_sigma_threshold 5; then
    echo "Launcher exit=0; finished: $(date -Is)"
else
    status=$?
    echo "Launcher exit=$status; failed: $(date -Is)"
    exit "$status"
fi
