#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-/home/user/lichengyuan/.conda_envs/feddistill310/bin/python}"
BASE_ID="${1:-12301}"
PROFILE="${2:-smoke}"
MODE="${3:-plan}"

if [[ ! "$BASE_ID" =~ ^[1-9][0-9]{0,8}$ ]] ||
   [[ "$PROFILE" != smoke && "$PROFILE" != full ]] ||
   [[ "$MODE" != plan && "$MODE" != run ]]; then
    echo "Usage: bash $0 [base_id] [smoke|full] [plan|run]" >&2
    exit 2
fi

CLIENTS=6
BYZANTINE=1
ROUNDS=2
START_ROUND=1
if [[ "$PROFILE" == full ]]; then
    CLIENTS=30
    BYZANTINE=9
    ROUNDS=70
    START_ROUND=15
fi
KRUM_F="$BYZANTINE"
NAMES=(clear_trimean cpa_plain_trimean clear_multikrum cpa_plain_multikrum clear_afa cpa_plain_afa)
ATTACKS=(NoAttack CPA NoAttack CPA NoAttack CPA)
DEFENCES=(TriMean TriMean Krum Krum AFA AFA)
END_ID=$((BASE_ID + 5))
SUMMARY="logs/${BASE_ID}_${END_ID}_cpa_traditional_${PROFILE}.summary.txt"

print_plan() {
    echo "CPA plain traditional defenses; profile=$PROFILE; rounds=$ROUNDS; clients=$CLIENTS; attack/probe start=$START_ROUND"
    echo "Six independent runs from initialization; no checkpoint reuse. Defenses operate from round 1; CPA starts at $START_ROUND."
    echo "FedDistill raw-logit aggregation; CIFAR-10/ResNet18; full STL-10 unlabeled public set."
    echo "Knowledge=honest_predictions (omniscient); covariance=stl10-cov.pt; class order=CIFAR-10 classifier."
    echo "CPA probability-to-logit epsilon=1e-12; attack fraction=1; HIPS=false."
    echo "SPC audit=true, filter=false in all runs; same scale/replace probes as 12101-12104."
    echo "TriMean ratio=.3; Multi-Krum f=$KRUM_F, selected=3, fallback=error in BOTH clean and attack groups."
    echo "AFA xi=2, delta_xi=.5, prior_alpha=prior_beta=3, block_threshold=.95; fresh history for each run."
    echo "Full reference: 12101 clean Mean=.7179, 12102 clean SPC=.7114, 12103 CPA Mean=.0442, 12104 CPA SPC=.5306."
    for i in "${!NAMES[@]}"; do
        printf '%s %s attack=%s defense=%s SPC_filter=false\n' "$((BASE_ID + i))" "${NAMES[$i]}" "${ATTACKS[$i]}" "${DEFENCES[$i]}"
    done
    echo "Smoke checks execution only. Every run has its own clean/attack setting; this is a logit-space adaptation."
}
if [[ "$MODE" == plan ]]; then
    print_plan
    exit 0
fi

cd "$PROJECT_DIR"
mkdir -p logs
[[ -x "$PYTHON" ]] || { echo "Python not found: $PYTHON; set PYTHON to your environment interpreter." >&2; exit 1; }
command -v flock >/dev/null || { echo "flock is required." >&2; exit 1; }
# Same lock as the existing matrix launchers, plus a process check for other schedulers.
exec 9>logs/.pcfdla_matrix.lock
flock -n 9 || { echo "Another matrix scheduler is active." >&2; exit 1; }
if pgrep -af '[p]ython[^ ]* .*main\.py|[p]ython[^ ]* .*train_clean_teacher\.py'; then
    echo "Another training process is active; stopping." >&2
    exit 1
fi
[[ ! -e "$SUMMARY" ]] || { echo "Existing summary: $SUMMARY; choose new run IDs." >&2; exit 1; }
shopt -s nullglob
for i in "${!NAMES[@]}"; do
    run_id=$((BASE_ID + i))
    for existing in logs/"${run_id}"_*; do
        echo "Run ID already exists: $existing; choose new run IDs." >&2
        exit 1
    done
done
export WANDB_MODE=offline

"$PYTHON" -m unittest discover -s tests -p 'test_cpa*.py' -v
"$PYTHON" main.py --help > /dev/null
"$PYTHON" - <<'PY'
from pathlib import Path
from types import SimpleNamespace

from torchvision.datasets import CIFAR10, STL10
from torchvision.datasets.utils import check_integrity
from byzantine.attacks import CPA

for train in (True, False):
    data = CIFAR10("datasets_pytorch/cifar10", train=train, download=False)
    assert len(data) == (50000 if train else 10000)
for name, md5 in STL10.train_list + STL10.test_list:
    path = Path("datasets_pytorch/stl10") / STL10.base_folder / name
    if not check_integrity(str(path), md5):
        raise RuntimeError(f"Missing/damaged STL-10 file: {path}")
CPA(config=SimpleNamespace(dataset="cifar10", public_ds="stl10", public_split_mode="legacy"),
    clients=[], runner_instance=SimpleNamespace(seed=20260805, n_classes=10))
print("CIFAR-10, STL-10 and CPA covariance checks passed; no downloads requested.")
PY

{
    echo "Started: $(date -Is)"
    print_plan
    echo "SGD momentum=.9 wd=.0001; batch=128; T=1; LR=(Linear, 0.1); local/KD epochs=1/1; seed=20260805."
    echo "No pretraining, client public KD, purifier, or mixed precision."
    sha256sum main.py byzantine/attacks.py byzantine/defences.py byzantine/cpa_info/stl10-cov.pt ssh/run_cpa_traditional_matrix.sh
} | tee "$SUMMARY"

for i in "${!NAMES[@]}"; do
    run_id=$((BASE_ID + i))
    attack="${ATTACKS[$i]}"
    log="logs/${run_id}_${NAMES[$i]}.log"
    n_byzantine="$BYZANTINE"
    [[ "$attack" == NoAttack ]] && n_byzantine=0
    echo "Starting $run_id: ${NAMES[$i]}; log=$log" | tee -a "$SUMMARY"
    echo "Started: $(date -Is)" > "$log"
    if "$PYTHON" -u main.py \
        --debug --run_id "$run_id" --seed 20260805 \
        --dataset cifar10 --arch ResNet18 --batch_size 128 \
        --strategy FedDistill --attack "$attack" --hips false \
        --cpa_cov_path none --cpa_logit_epsilon 1e-12 --sample_attack_frac 1 \
        --defence "${DEFENCES[$i]}" --memory_method none \
        --trimean_trim_ratio .3 --krum_f "$KRUM_F" --krum_num_selected 3 --krum_fallback error \
        --afa_xi 2 --afa_delta_xi .5 --afa_prior_alpha 3 --afa_prior_beta 3 --afa_block_threshold .95 \
        --n_clients "$CLIENTS" --n_byzantine_clients "$n_byzantine" \
        --n_total_local_epochs "$ROUNDS" --n_communications "$ROUNDS" \
        --n_server_epochs_per_round 1 --optimizer SGD --momentum .9 --weight_decay .0001 \
        --server_lr "(Linear, 0.1)" --client_lr "(Linear, 0.1)" \
        --restart_client_lr none --reinit_server none --server_init_ckpt none \
        --distill_temperature 1 --attack_start_round "$START_ROUND" \
        --use_amp false --use_diffusion_purifier false --evaluate_backdoor_asr false \
        --public_split_mode legacy --public_ds stl10 --public_ds_fraction none \
        --use_probe true --probe_mode replace --probe_type scale --probe_score hybrid \
        --probe_base_count 20 --probe_scales 1.0,1.2,1.5,2.0 \
        --probe_seed 0 --probe_start_round "$START_ROUND" \
        --probe_core_sigma_audit true --probe_core_sigma_filter false \
        --probe_core_sigma_threshold 5 \
        >> "$log" 2>&1; then
        status=0
    else
        status=$?
    fi
    echo "Exit code: $status; ended: $(date -Is)" >> "$log"
    {
        echo "Run $run_id ${NAMES[$i]} exit=$status"
        grep -F 'Final accuracy summary' "$log" | tail -n 1 || true
        grep -F '[CPA]' "$log" | tail -n 1 || true
        grep -E '\[TriMean\]|\[Multi-Krum\]|\[Krum selection audit\]|\[AFA retention audit\]' "$log" | tail -n 2 || true
    } | tee -a "$SUMMARY"
    if [[ "$status" -ne 0 ]]; then
        tail -n 25 "$log"
        exit "$status"
    fi
done
echo "All six runs completed: $(date -Is)" | tee -a "$SUMMARY"
