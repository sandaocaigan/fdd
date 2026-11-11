#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-/home/user/lichengyuan/.conda_envs/feddistill310/bin/python}"
BASE_ID="${1:-11701}"
PROFILE="${2:-full}"
MODE="${3:-plan}"
if [[ ! "$BASE_ID" =~ ^[1-9][0-9]{0,8}$ ]] ||
   [[ "$PROFILE" != full && "$PROFILE" != smoke ]] ||
   [[ "$MODE" != run && "$MODE" != plan ]]; then
    echo "Usage: bash $0 [base_id] [full|smoke] [plan|run]" >&2
    exit 2
fi

CLIENTS=30; BYZANTINE=9; ATTACK_START=15; BRANCH_ROUNDS=56; SAMPLES=2048
CHECKPOINT="${PAIR_CHECKPOINT:-checkpoints/poisonedfl_pair_11111_round14.pt}"
if [[ "$PROFILE" == smoke ]]; then
    CLIENTS=6; BYZANTINE=1; ATTACK_START=2; BRANCH_ROUNDS=2; SAMPLES=128
    CHECKPOINT="${PAIR_CHECKPOINT:-checkpoints/poisonedfl_traditional_${BASE_ID}_round1.pt}"
fi
LAST_ID=$((BASE_ID + 6))
WARMUP=$((ATTACK_START - 1))
SUMMARY="logs/${BASE_ID}_${LAST_ID}_poisonedfl_paired.summary.txt"
NAMES=(warmup clear_trimean strength10_trimean clear_multikrum strength10_multikrum clear_afa strength10_afa)

print_plan() {
    echo "PoisonedFL-inspired logits strength10 vs traditional defences; profile=$PROFILE; seed=20260805"
    echo "CIFAR-10/ResNet18; clients=$CLIENTS; malicious=$BYZANTINE in attacks; LR horizon=70"
    echo "Mean warmup checkpoint=$CHECKPOINT; warmup=$WARMUP; each branch=$BRANCH_ROUNDS rounds"
    echo "SPC audit=true; SPC filter=false; fixed direction; adaptive=false"
    echo "TriMean ratio=0.3; Multi-Krum f=$BYZANTINE selected=3 fallback=error in clean AND attacked branches"
    echo "AFA xi=2 delta_xi=0.5 prior_alpha=3 prior_beta=3 block_threshold=0.95; fresh history per branch"
    echo "Robust aggregation begins after the shared warmup; warmup uses Mean."
    for i in "${!NAMES[@]}"; do echo "$((BASE_ID + i)) ${NAMES[$i]}"; done
    echo "delta_vs_clear_pp compares attacked vs clean under the SAME defence at the SAME round."
    if [[ "$PROFILE" == full ]]; then
        echo "Reuse existing references: 11302/11305 Mean; 11502/11503 SPC. No repeat training."
    fi
}
if [[ "$MODE" == plan ]]; then print_plan; exit 0; fi
cd "$PROJECT_DIR"
mkdir -p logs
[[ -x "$PYTHON" ]] || { echo "Python not found: $PYTHON" >&2; exit 1; }
if [[ "$PROFILE" == full && ! -f "$CHECKPOINT" ]]; then
    echo "Missing shared round-14 checkpoint: $CHECKPOINT. Set PAIR_CHECKPOINT to the original compatible checkpoint." >&2
    exit 1
fi
command -v flock >/dev/null || { echo "flock is required." >&2; exit 1; }
exec 9>logs/.pcfdla_matrix.lock
flock -n 9 || { echo "Another experiment scheduler is active." >&2; exit 1; }
if pgrep -af '[p]ython[^ ]* .*main\.py|[p]ython[^ ]* .*train_clean_teacher\.py'; then
    echo "Another training process is active; refusing to start." >&2
    exit 1
fi
[[ ! -e "$SUMMARY" ]] || { echo "Existing summary: $SUMMARY; use fresh IDs." >&2; exit 1; }
shopt -s nullglob
for ((id=BASE_ID; id<=LAST_ID; id++)); do
    for existing in logs/"${id}"_*; do
        [[ "$existing" == *.launcher.log || "$existing" == *.launcher.pid ]] && continue
        echo "Run ID already has output: $existing; choose fresh IDs." >&2
        exit 1
    done
done
export WANDB_MODE=offline
print_plan
"$PYTHON" -m unittest discover -s tests -v
"$PYTHON" main.py --help | grep -- '--poisonedfl_pair_traditional' >/dev/null
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
echo "Traditional-defence experiment started: $(date -Is)"
sha256sum main.py runners/PoisonedFLPairRunner.py runners/ScratchRunner.py \
    byzantine/attacks.py byzantine/poisonedfl.py byzantine/defences.py \
    ssh/run_poisonedfl_traditional.sh
# Leave robust hyperparameters at the original defaults during checkpoint
# verification; the paired runner installs the registered settings per branch.
if "$PYTHON" -u main.py --debug --run_id "$BASE_ID" --seed 20260805 \
    --dataset cifar10 --arch ResNet18 --batch_size 128 \
    --strategy FedDistill --attack PoisonedFL --defence NoDefence --memory_method none \
    --n_clients "$CLIENTS" --n_byzantine_clients "$BYZANTINE" \
    --n_total_local_epochs 70 --n_communications 70 --n_server_epochs_per_round 1 \
    --optimizer SGD --momentum 0.9 --weight_decay 0.0001 \
    --server_lr "(Linear, 0.1)" --client_lr "(Linear, 0.1)" \
    --restart_client_lr false --reinit_server false --warm_restarts false \
    --server_init_ckpt none --client_early_stopping false --server_early_stopping false \
    --distill_temperature 1 --attack_start_round "$ATTACK_START" \
    --poisonedfl_strength 10 --poisonedfl_direction fixed --poisonedfl_adaptive false \
    --poisonedfl_min_strength 0.01 --poisonedfl_max_strength 10 \
    --poisonedfl_pair_rounds "$BRANCH_ROUNDS" --poisonedfl_pair_samples "$SAMPLES" \
    --poisonedfl_pair_checkpoint "$CHECKPOINT" \
    --poisonedfl_pair_spc false --poisonedfl_pair_spc_only false --poisonedfl_pair_traditional true \
    --use_amp false --use_diffusion_purifier false --evaluate_backdoor_asr false \
    --public_split_mode legacy --public_ds stl10 --public_ds_fraction none \
    --use_probe true --probe_mode replace --probe_type scale --probe_score hybrid \
    --probe_base_count 20 --probe_scales 1.0,1.2,1.5,2.0 --probe_seed 0 \
    --probe_start_round "$ATTACK_START" --probe_core_sigma_audit true \
    --probe_core_sigma_filter false --probe_core_sigma_threshold 5; then
    echo "Launcher exit=0; finished: $(date -Is)"
else
    status=$?
    echo "Launcher exit=$status; failed: $(date -Is)" >&2
    exit "$status"
fi
