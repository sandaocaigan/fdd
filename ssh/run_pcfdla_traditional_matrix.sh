#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-/home/user/lichengyuan/.conda_envs/feddistill310/bin/python}"
BASE_ID="${1:-11021}"
START_AT="${2:-clear_trimean}"
MODE="${3:-run}"
NAMES=(clear_trimean pcfdla_s10_trimean clear_multikrum pcfdla_s10_multikrum clear_afa pcfdla_s10_afa)
ATTACKS=(NoAttack PCFDLA NoAttack PCFDLA NoAttack PCFDLA)
DEFENCES=(TriMean TriMean Krum Krum AFA AFA)

if [[ ! "$BASE_ID" =~ ^[1-9][0-9]{0,8}$ ]] || [[ "$MODE" != run && "$MODE" != plan ]]; then
    echo "Usage: bash $0 [base_run_id] [start_experiment] [run|plan]"
    exit 2
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
SUMMARY="logs/${BASE_ID}_${END_ID}_pcfdla_traditional.summary.txt"
if [[ "$MODE" == plan ]]; then
    echo "PCFDLA S=10; FedDistill; ResNet18; no pretraining; 30 clients; 70 rounds; seed=20260805"
    echo "Attack starts at 15; attack groups have 9 malicious clients; clean groups have 0."
    echo "All groups use identical STL-10/probes; audit=true, SPC filter=false."
    echo "TriMean ratio=0.3 (9 per tail); Multi-Krum f=9 selected=3 in BOTH clean and attacked runs."
    for ((i=START_INDEX; i<${#NAMES[@]}; i++)); do
        printf '%s %s attack=%s defence=%s\n' \
            "$((BASE_ID + i - START_INDEX))" "${NAMES[$i]}" "${ATTACKS[$i]}" "${DEFENCES[$i]}"
    done
    exit 0
fi

cd "$PROJECT_DIR"
mkdir -p logs
[[ -x "$PYTHON" ]] || { echo "Python not found: $PYTHON"; exit 1; }
command -v flock >/dev/null || { echo "flock is required."; exit 1; }
# Share the lock with the PCFDLA/SPC scheduler, including its preflight phase.
exec 9>logs/.pcfdla_matrix.lock
flock -n 9 || { echo "Another PCFDLA scheduler is active."; exit 1; }
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
"$PYTHON" -m unittest discover -s tests -p 'test_pcfdla*.py' -v
"$PYTHON" main.py --help | grep -- '--pcfdla_peak' >/dev/null
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
print("CIFAR-10 and STL-10 integrity checks passed; no downloads requested.")
PY

{
    echo "PCFDLA traditional-defence comparison started: $(date -Is)"
    echo "Run IDs: $BASE_ID-$END_ID; start at: $START_AT"
    echo "Project: $PROJECT_DIR; Python: $PYTHON"
    echo "FedDistill, ResNet18, random initialization, no checkpoint, no client public KD"
    echo "CIFAR-10 private=47500 validation=2500; full STL-10 public=100000"
    echo "30 clients; 9 malicious for PCFDLA, 0 for NoAttack; seed=20260805"
    echo "70 rounds; attack/probe start=15; raw-logit peak=10"
    echo "Batch=128; T=1; SGD momentum=0.9 weight_decay=0.0001"
    echo "Server/client LR=(Linear, 0.1); private/server epochs per round=1/1"
    echo "Identical probes: scale/replace, 20 groups, scales=1.0,1.2,1.5,2.0, seed=0"
    echo "SPC audit=true, filter=false for every new run; purifier=false, AMP=false"
    echo "TriMean: ratio=0.3; Multi-Krum: f=9 selected=3 fallback=error (clean AND attacked)"
    echo "AFA: xi=2 delta_xi=0.5 prior_alpha=3 prior_beta=3 block_threshold=0.95"
    echo "Historical table used TriMean ratio=0.2; this registered comparison uses 0.3."
    echo "Existing same-protocol references: 11011 clean Mean, 11012 clean SPC, 11017 PCFDLA Mean, 11018 PCFDLA SPC."
    echo "These are logit-space implementations of traditional model aggregation rules."
    sha256sum main.py byzantine/attacks.py byzantine/defences.py ssh/run_pcfdla_traditional_matrix.sh
} | tee "$SUMMARY"

for ((i=START_INDEX; i<${#NAMES[@]}; i++)); do
    run_id=$((BASE_ID + i - START_INDEX))
    name="${NAMES[$i]}"
    attack="${ATTACKS[$i]}"
    defence="${DEFENCES[$i]}"
    n_byzantine=0
    [[ "$attack" == PCFDLA ]] && n_byzantine=9
    log="logs/${run_id}_${name}.log"
    {
        echo "Start: $(date -Is)"
        echo "Run ID: $run_id; attack=$attack; peak=10; defence=$defence; SPC filter=false"
    } > "$log"
    echo "Starting run $run_id: $name; progress: $log" | tee -a "$SUMMARY"
    if "$PYTHON" -u main.py \
        --debug --run_id "$run_id" --seed 20260805 \
        --dataset cifar10 --arch ResNet18 --batch_size 128 \
        --strategy FedDistill --attack "$attack" --pcfdla_peak 10 \
        --defence "$defence" --memory_method none \
        --trimean_trim_ratio 0.3 --krum_f 9 --krum_num_selected 3 --krum_fallback error \
        --afa_xi 2 --afa_delta_xi 0.5 --afa_prior_alpha 3 --afa_prior_beta 3 --afa_block_threshold 0.95 \
        --n_clients 30 --n_byzantine_clients "$n_byzantine" \
        --n_total_local_epochs 70 --n_communications 70 --n_server_epochs_per_round 1 \
        --optimizer SGD --momentum 0.9 --weight_decay 0.0001 \
        --server_lr "(Linear, 0.1)" --client_lr "(Linear, 0.1)" \
        --restart_client_lr none --reinit_server none --server_init_ckpt none \
        --distill_temperature 1 --attack_start_round 15 \
        --use_amp false --use_diffusion_purifier false \
        --public_split_mode legacy --public_ds stl10 --public_ds_fraction none \
        --use_probe true --probe_mode replace --probe_type scale --probe_score hybrid \
        --probe_base_count 20 --probe_scales 1.0,1.2,1.5,2.0 --probe_seed 0 --probe_start_round 15 \
        --probe_core_sigma_audit true --probe_core_sigma_filter false --probe_core_sigma_threshold 5 \
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
        echo "Run $run_id: attack=$attack peak=10 defence=$defence exit=$status"
        grep -F 'Final accuracy summary' "$log" | tail -n 1 || true
        grep -F '[PCFDLA]' "$log" | tail -n 1 || true
        grep -E '\[TriMean\]|\[Multi-Krum\]|\[Krum selection audit\]|\[AFA retention audit\]' "$log" | tail -n 2 || true
    } | tee -a "$SUMMARY"
    if [[ "$status" -ne 0 ]]; then
        tail -n 25 "$log"
        exit "$status"
    fi
done
echo "All PCFDLA traditional-defence experiments finished: $(date -Is)" | tee -a "$SUMMARY"
