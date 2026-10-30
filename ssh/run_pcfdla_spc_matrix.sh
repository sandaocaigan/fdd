#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-/home/user/lichengyuan/.conda_envs/feddistill310/bin/python}"
BASE_ID="${1:-11001}"
PROFILE="${2:-smoke}"
MODE="${3:-run}"

if [[ ! "$BASE_ID" =~ ^[1-9][0-9]{0,8}$ ]] ||
   [[ "$PROFILE" != smoke && "$PROFILE" != full ]] ||
   [[ "$MODE" != run && "$MODE" != plan ]]; then
    echo "Usage: bash $0 [positive_base_run_id] [smoke|full] [run|plan]"
    exit 2
fi

NAMES=(pcfdla_s5_no_filter pcfdla_s5_spc)
ATTACKS=(PCFDLA PCFDLA)
PEAKS=(5 5)
FILTERS=(false true)
CLIENTS=4
BYZANTINE=1
ROUNDS=2
START_ROUND=1
if [[ "$PROFILE" == full ]]; then
    NAMES=(clear_no_filter clear_spc pcfdla_s1_no_filter pcfdla_s1_spc
           pcfdla_s5_no_filter pcfdla_s5_spc pcfdla_s10_no_filter pcfdla_s10_spc)
    ATTACKS=(NoAttack NoAttack PCFDLA PCFDLA PCFDLA PCFDLA PCFDLA PCFDLA)
    PEAKS=(5 5 1 1 5 5 10 10)
    FILTERS=(false true false true false true false true)
    CLIENTS=30
    BYZANTINE=9
    ROUNDS=70
    START_ROUND=15
fi
END_ID=$((BASE_ID + ${#NAMES[@]} - 1))
SUMMARY="logs/${BASE_ID}_${END_ID}_pcfdla_${PROFILE}.summary.txt"

if [[ "$MODE" == plan ]]; then
    echo "Profile=$PROFILE; rounds=$ROUNDS; clients=$CLIENTS; attack/probe start=$START_ROUND"
    echo "FedDistill; ResNet18; no pretraining; CIFAR-10 private; full STL-10 unlabeled public"
    echo "All groups use identical probe settings; only filtering is toggled. Peaks are adaptation settings."
    for i in "${!NAMES[@]}"; do
        printf '%s %s attack=%s peak=%s filter=%s\n' \
            "$((BASE_ID + i))" "${NAMES[$i]}" "${ATTACKS[$i]}" "${PEAKS[$i]}" "${FILTERS[$i]}"
    done
    exit 0
fi

cd "$PROJECT_DIR"
mkdir -p logs
[[ -x "$PYTHON" ]] || { echo "Python not found: $PYTHON"; exit 1; }
command -v flock >/dev/null || { echo "flock is required."; exit 1; }
exec 9>logs/.pcfdla_matrix.lock
flock -n 9 || { echo "Another PCFDLA scheduler is active."; exit 1; }
if pgrep -af '[p]ython[^ ]* .*main\.py|[p]ython[^ ]* .*train_clean_teacher\.py'; then
    echo "Another training process is active; refusing to start."
    exit 1
fi
[[ ! -e "$SUMMARY" ]] || { echo "Existing summary: $SUMMARY"; exit 1; }
shopt -s nullglob
for i in "${!NAMES[@]}"; do
    run_id=$((BASE_ID + i))
    for existing in logs/"${run_id}"_*.log; do
        # A launcher may begin with the first run ID; it is not a training log.
        [[ "$existing" == *.launcher.log ]] && continue
        echo "Run ID already has a log: $existing; choose a fresh base ID."
        exit 1
    done
done
export WANDB_MODE=offline

echo "Checking PCFDLA and CLI support."
"$PYTHON" -m unittest discover -s tests -p test_pcfdla.py -v
"$PYTHON" main.py --help | grep -- '--pcfdla_peak' >/dev/null
"$PYTHON" - <<'PY'
from pathlib import Path
from torchvision.datasets import CIFAR10, STL10
from torchvision.datasets.utils import check_integrity

root = Path("datasets_pytorch/cifar10")
for train in (True, False):
    dataset = CIFAR10(str(root), train=train, download=False)
    assert len(dataset) == (50000 if train else 10000)
root = Path("datasets_pytorch/stl10")
for name, md5 in STL10.train_list + STL10.test_list:
    path = root / STL10.base_folder / name
    if not check_integrity(str(path), md5):
        raise RuntimeError(f"Missing or damaged STL-10 file: {path}")
print("CIFAR-10 and STL-10 integrity checks passed; no downloads requested.")
PY

{
    echo "PCFDLA $PROFILE started: $(date -Is)"
    echo "Project: $PROJECT_DIR; Python: $PYTHON; Run IDs: $BASE_ID-$END_ID"
    echo "FedDistill; ResNet18; random initialization; no client public KD"
    echo "CIFAR-10 private=47500, validation=2500; STL-10 public=100000"
    echo "Rounds=$ROUNDS; clients=$CLIENTS; attack/probe start=$START_ROUND"
    echo "SGD momentum=0.9 weight_decay=0.0001; batch=128; T=1"
    echo "Server/client LR=(Linear, 0.1); private/server epochs per round=1/1"
    echo "Probe replace/scale, 20 groups x 4 variants, seed=0; SPC threshold=5"
    echo "Identical probe inputs and audit settings in filter-on and filter-off groups."
    echo "Peak S=1/5/10 are this project's trial values, not verified paper constants."
    echo "Smoke validates execution only; it does not measure convergence or defense efficacy."
} | tee "$SUMMARY"

for i in "${!NAMES[@]}"; do
    run_id=$((BASE_ID + i))
    attack="${ATTACKS[$i]}"
    peak="${PEAKS[$i]}"
    filter="${FILTERS[$i]}"
    log="logs/${run_id}_${NAMES[$i]}.log"
    n_byzantine="$BYZANTINE"
    [[ "$attack" == NoAttack ]] && n_byzantine=0
    {
        echo "Start: $(date -Is)"
        echo "Run ID: $run_id; attack=$attack; peak=$peak; filter=$filter"
    } > "$log"
    echo "Starting run $run_id: ${NAMES[$i]}; detailed progress: $log" | tee -a "$SUMMARY"
    if "$PYTHON" -u main.py \
        --debug --run_id "$run_id" --seed 20260805 \
        --dataset cifar10 --arch ResNet18 --batch_size 128 \
        --strategy FedDistill --attack "$attack" --pcfdla_peak "$peak" \
        --defence NoDefence --memory_method none \
        --n_clients "$CLIENTS" --n_byzantine_clients "$n_byzantine" \
        --n_total_local_epochs "$ROUNDS" --n_communications "$ROUNDS" \
        --n_server_epochs_per_round 1 --optimizer SGD --momentum 0.9 --weight_decay 0.0001 \
        --server_lr "(Linear, 0.1)" --client_lr "(Linear, 0.1)" \
        --restart_client_lr none --reinit_server none --server_init_ckpt none \
        --distill_temperature 1 --attack_start_round "$START_ROUND" \
        --use_amp false --use_diffusion_purifier false \
        --public_split_mode legacy --public_ds stl10 --public_ds_fraction none \
        --use_probe true --probe_mode replace --probe_type scale --probe_score hybrid \
        --probe_base_count 20 --probe_scales 1.0,1.2,1.5,2.0 \
        --probe_seed 0 --probe_start_round "$START_ROUND" \
        --probe_core_sigma_audit true --probe_core_sigma_filter "$filter" \
        --probe_core_sigma_threshold 5 \
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
        echo "Run $run_id: attack=$attack peak=$peak filter=$filter exit=$status"
        grep -F 'Final accuracy summary' "$log" | tail -n 1 || true
        grep -F '[PCFDLA]' "$log" | tail -n 1 || true
        grep -F '[Probe Core-Sigma Filter]' "$log" | tail -n 1 || true
    } | tee -a "$SUMMARY"
    if [[ "$status" -ne 0 ]]; then
        tail -n 25 "$log"
        exit "$status"
    fi
done
echo "All PCFDLA experiments finished: $(date -Is)" | tee -a "$SUMMARY"
