#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-/home/user/lichengyuan/.conda_envs/feddistill310/bin/python}"
BASE_ID="${1:-11031}"
PROFILE="${2:-smoke}"
MODE="${3:-run}"
START_AT="${4:-clear_no_filter}"
STRENGTH="${POISONEDFL_STRENGTH:-1.0}"
NAMES=(clear_no_filter clear_spc fixed_no_filter fresh_no_filter adaptive_no_filter fixed_spc adaptive_spc)
ATTACKS=(NoAttack NoAttack PoisonedFL PoisonedFL PoisonedFL PoisonedFL PoisonedFL)
DIRECTIONS=(fixed fixed fixed fresh fixed fixed fixed)
ADAPTIVE=(false false false false true false true)
FILTERS=(false true false false false true true)
CLIENTS=4
BYZANTINE=1
ROUNDS=2
START_ROUND=1
FEEDBACK_INTERVAL=1

if [[ ! "$BASE_ID" =~ ^[1-9][0-9]{0,8}$ ]] ||
   [[ "$PROFILE" != smoke && "$PROFILE" != full ]] ||
   [[ "$MODE" != run && "$MODE" != plan ]]; then
    echo "Usage: bash $0 [base_id] [smoke|full] [run|plan] [start_experiment]"
    exit 2
fi
if [[ "$PROFILE" == full ]]; then
    CLIENTS=30
    BYZANTINE=9
    ROUNDS=70
    START_ROUND=15
    FEEDBACK_INTERVAL=5
fi
START_INDEX=-1
for i in "${!NAMES[@]}"; do
    if [[ "${NAMES[$i]}" == "$START_AT" ]]; then START_INDEX="$i"; break; fi
done
if [[ "$START_INDEX" -lt 0 ]]; then
    echo "Unknown start experiment: $START_AT; choose: ${NAMES[*]}"
    exit 2
fi
END_ID=$((BASE_ID + ${#NAMES[@]} - START_INDEX - 1))
SUMMARY="logs/${BASE_ID}_${END_ID}_poisonedfl_${PROFILE}.summary.txt"
if [[ "$MODE" == plan ]]; then
    echo "PoisonedFL-inspired logits adaptation; NOT the original parameter-upload algorithm."
    echo "Profile=$PROFILE; clients=$CLIENTS; malicious=$BYZANTINE in attacks; rounds=$ROUNDS; start=$START_ROUND"
    echo "Strength=$STRENGTH raw-logit RMS; bounds=0.01..10; feedback interval=$FEEDBACK_INTERVAL"
    echo "Shared public images/probes; SPC audit=true; no client public KD or pretraining."
    for ((i=START_INDEX; i<${#NAMES[@]}; i++)); do
        printf '%s %s attack=%s direction=%s adaptive=%s filter=%s\n' \
            "$((BASE_ID + i - START_INDEX))" "${NAMES[$i]}" "${ATTACKS[$i]}" \
            "${DIRECTIONS[$i]}" "${ADAPTIVE[$i]}" "${FILTERS[$i]}"
    done
    exit 0
fi

cd "$PROJECT_DIR"
mkdir -p logs
[[ -x "$PYTHON" ]] || { echo "Python not found: $PYTHON"; exit 1; }
command -v flock >/dev/null || { echo "flock is required."; exit 1; }
# The shared lock also protects the older PCFDLA launchers during preflight.
exec 9>logs/.pcfdla_matrix.lock
flock -n 9 || { echo "Another attack scheduler is active."; exit 1; }
if pgrep -af '[p]ython[^ ]* .*main\.py|[p]ython[^ ]* .*train_clean_teacher\.py'; then
    echo "Another training process is active; refusing to start."
    exit 1
fi
[[ ! -e "$SUMMARY" ]] || { echo "Existing summary: $SUMMARY"; exit 1; }
shopt -s nullglob
for ((i=START_INDEX; i<${#NAMES[@]}; i++)); do
    run_id=$((BASE_ID + i - START_INDEX))
    for existing in logs/"${run_id}"_*.log; do
        [[ "$existing" == *.launcher.log ]] && continue
        echo "Run ID already has a log: $existing; choose fresh IDs."
        exit 1
    done
done
export WANDB_MODE=offline
"$PYTHON" -m unittest discover -s tests -v
"$PYTHON" main.py --help | grep -- '--poisonedfl_strength' >/dev/null
"$PYTHON" - "$STRENGTH" <<'PY'
import sys
from pathlib import Path
from torchvision.datasets import CIFAR10, STL10
from torchvision.datasets.utils import check_integrity
from byzantine.poisonedfl import PoisonedFLLogitController

PoisonedFLLogitController(seed=20260805, strength=float(sys.argv[1]))
for train in (True, False):
    dataset = CIFAR10("datasets_pytorch/cifar10", train=train, download=False)
    assert len(dataset) == (50000 if train else 10000)
for name, md5 in STL10.train_list + STL10.test_list:
    path = Path("datasets_pytorch/stl10") / STL10.base_folder / name
    if not check_integrity(str(path), md5):
        raise RuntimeError(f"Missing or damaged STL-10 file: {path}")
print("CIFAR-10/STL-10 checks passed without downloads.")
PY

{
    echo "PoisonedFL $PROFILE started: $(date -Is)"
    echo "Run IDs: $BASE_ID-$END_ID; project=$PROJECT_DIR; Python=$PYTHON"
    echo "Output-space adaptation; not an exact reproduction of CVPR 2025 PoisonedFL."
    echo "CIFAR-10 private=47500 validation=2500; STL-10 public=100000; seed=20260805"
    echo "FedDistill; random ResNet18; no checkpoint; no client public KD"
    echo "Rounds=$ROUNDS; clients=$CLIENTS; malicious=$BYZANTINE for attacks; attack/probe start=$START_ROUND"
    echo "SGD momentum=0.9 wd=0.0001; batch=128; T=1; server/client LR=(Linear, 0.1); epochs=1/1"
    echo "Strength=$STRENGTH; centered unit-RMS sample directions; fixed vs fresh; bounds=0.01..10"
    echo "Adaptive feedback: interval=$FEEDBACK_INTERVAL target=0.55 factor=1.25; broadcast-server outputs only"
    echo "Probes: replace/scale, 20 groups, scales=1,1.2,1.5,2; seed=0; audit=true; threshold=5"
    echo "Smoke validates execution and diagnostics only, not convergence or attack efficacy."
    sha256sum main.py byzantine/attacks.py byzantine/poisonedfl.py runners/ScratchRunner.py \
        probes/probe_auditor.py ssh/run_poisonedfl_matrix.sh
} | tee "$SUMMARY"

for ((i=START_INDEX; i<${#NAMES[@]}; i++)); do
    run_id=$((BASE_ID + i - START_INDEX))
    log="logs/${run_id}_poisonedfl_${NAMES[$i]}.log"
    n_byzantine="$BYZANTINE"
    [[ "${ATTACKS[$i]}" == NoAttack ]] && n_byzantine=0
    {
        echo "Start: $(date -Is)"
        echo "Run ID: $run_id; condition=${NAMES[$i]}; strength=$STRENGTH"
    } > "$log"
    echo "Starting run $run_id: ${NAMES[$i]}; progress: $log" | tee -a "$SUMMARY"
    if "$PYTHON" -u main.py \
        --debug --run_id "$run_id" --seed 20260805 \
        --dataset cifar10 --arch ResNet18 --batch_size 128 \
        --strategy FedDistill --attack "${ATTACKS[$i]}" --defence NoDefence --memory_method none \
        --poisonedfl_strength "$STRENGTH" --poisonedfl_direction "${DIRECTIONS[$i]}" \
        --poisonedfl_adaptive "${ADAPTIVE[$i]}" --poisonedfl_min_strength 0.01 \
        --poisonedfl_max_strength 10 --poisonedfl_feedback_interval "$FEEDBACK_INTERVAL" \
        --poisonedfl_feedback_target 0.55 --poisonedfl_adjust_factor 1.25 \
        --n_clients "$CLIENTS" --n_byzantine_clients "$n_byzantine" \
        --n_total_local_epochs "$ROUNDS" --n_communications "$ROUNDS" \
        --n_server_epochs_per_round 1 --optimizer SGD --momentum 0.9 --weight_decay 0.0001 \
        --server_lr "(Linear, 0.1)" --client_lr "(Linear, 0.1)" \
        --restart_client_lr none --reinit_server none --server_init_ckpt none \
        --distill_temperature 1 --attack_start_round "$START_ROUND" \
        --use_amp false --use_diffusion_purifier false \
        --public_split_mode legacy --public_ds stl10 --public_ds_fraction none \
        --use_probe true --probe_mode replace --probe_type scale --probe_score hybrid \
        --probe_base_count 20 --probe_scales 1.0,1.2,1.5,2.0 --probe_seed 0 \
        --probe_start_round "$START_ROUND" --probe_core_sigma_audit true \
        --probe_core_sigma_filter "${FILTERS[$i]}" --probe_core_sigma_threshold 5 \
        >> "$log" 2>&1; then
        status=0
    else
        status=$?
    fi
    {
        echo "Exit code: $status"
        echo "End: $(date -Is)"
    } >> "$log"
    {
        echo "Run $run_id: condition=${NAMES[$i]} exit=$status"
        grep -F 'Final accuracy summary' "$log" | tail -n 1 || true
        grep -F '[PoisonedFL inspired]' "$log" | tail -n 2 || true
        grep -F '[Probe Core-Sigma Filter]' "$log" | tail -n 1 || true
    } | tee -a "$SUMMARY"
    if [[ "$status" -ne 0 ]]; then tail -n 25 "$log"; exit "$status"; fi
done
echo "All PoisonedFL experiments finished: $(date -Is)" | tee -a "$SUMMARY"
