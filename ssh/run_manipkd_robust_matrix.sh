#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-/home/user/lichengyuan/.conda_envs/feddistill310/bin/python}"
CHECKPOINT="checkpoints/cifar10_resnet18_public5000_seed20260805.pt"
MANIFEST="splits/cifar10_public_500x10_seed20260805.json"
BASE_ID="${1:-10523}"
START_AT="${2:-clear_mean}"
MODE="${3:-run}"

EXPERIMENT_NAMES=(
    clear_mean manipkd_mean clear_krum manipkd_krum
    clear_trimean manipkd_trimean clear_afa manipkd_afa
    clear_spc7 manipkd_spc7
)
ATTACKS=(
    NoAttack ManipulatingKD NoAttack ManipulatingKD
    NoAttack ManipulatingKD NoAttack ManipulatingKD
    NoAttack ManipulatingKD
)
DEFENCES=(
    NoDefence NoDefence Krum Krum TriMean TriMean
    AFA AFA NoDefence NoDefence
)

if [[ ! "$BASE_ID" =~ ^[0-9]+$ ]]; then
    echo "Usage: $0 [base_run_id] [start_experiment] [run|plan]"
    exit 2
fi
if [ "$MODE" != run ] && [ "$MODE" != plan ]; then
    echo "Mode must be run or plan: $MODE"
    exit 2
fi

START_INDEX=-1
for i in "${!EXPERIMENT_NAMES[@]}"; do
    if [ "${EXPERIMENT_NAMES[$i]}" = "$START_AT" ]; then
        START_INDEX="$i"
        break
    fi
done
if [ "$START_INDEX" -lt 0 ]; then
    echo "Unknown start experiment: $START_AT"
    echo "Choose one of: ${EXPERIMENT_NAMES[*]}"
    exit 2
fi

END_ID=$((BASE_ID + ${#EXPERIMENT_NAMES[@]} - START_INDEX - 1))
SUMMARY="logs/${BASE_ID}_${END_ID}_manipkd_robust_matrix.summary.txt"

if [ "$MODE" = plan ]; then
    for ((i = START_INDEX; i < ${#EXPERIMENT_NAMES[@]}; i++)); do
        printf '%s %s attack=%s defence=%s\n' \
            "$((BASE_ID + i - START_INDEX))" "${EXPERIMENT_NAMES[$i]}" \
            "${ATTACKS[$i]}" "${DEFENCES[$i]}"
    done
    exit 0
fi

cd "$PROJECT_DIR"
mkdir -p logs
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ ! -x "$PYTHON" ] || [ ! -f "$CHECKPOINT" ] || [ ! -f "$MANIFEST" ]; then
    echo "Missing Python, server checkpoint, or public split manifest."
    exit 1
fi
if pgrep -af '[p]ython.*main.py|[p]ython.*train_clean_teacher.py' >/dev/null; then
    echo "Another training process is active; refusing to start."
    exit 1
fi

for ((i = START_INDEX; i < ${#EXPERIMENT_NAMES[@]}; i++)); do
    log="logs/$((BASE_ID + i - START_INDEX))_${EXPERIMENT_NAMES[$i]}.log"
    if [ -e "$log" ]; then
        echo "Existing result would be overwritten: $log"
        exit 1
    fi
done
if [ -e "$SUMMARY" ]; then
    echo "Existing result would be overwritten: $SUMMARY"
    exit 1
fi

"$PYTHON" -m unittest \
    tests.test_manipulatingkd_history \
    tests.test_server_pretraining \
    tests.test_teacher_public_split \
    tests.test_data_splits -v

{
    echo "ManipulatingKD robust aggregation matrix started: $(date)"
    echo "Base Run ID: $BASE_ID"
    echo "Start experiment: $START_AT"
    echo "Scheduled Run IDs: $BASE_ID-$END_ID"
    echo "Public split: $MANIFEST"
    echo "Server checkpoint: $CHECKPOINT"
    echo "Seed: 20260805; attack starts at round 15; SPC threshold: 7.0 when enabled"
    echo "P-L estimator: mean_surrogate; history: 5 rounds"
    echo
} > "$SUMMARY"

run_exp() {
    local run_id="$1"
    local name="$2"
    local attack="$3"
    local defence="$4"
    local use_spc="${5:-false}"
    local log="logs/${run_id}_${name}.log"
    local n_byzantine=0
    local defence_args=()
    local attack_args=()
    local probe_args=(
        --use_probe false
        --probe_core_sigma_audit false
        --probe_core_sigma_filter false
    )
    local status

    if [ "$attack" = "ManipulatingKD" ]; then
        n_byzantine=9
        attack_args=(
            --manipkd_formulation paper
            --manipkd_knowledge partial_local
            --manipkd_pl_estimator mean_surrogate
            --manipkd_history_rounds 5
            --manipkd_noise_std 0.10
            --manipkd_lr 0.50
            --manipkd_steps 60
            --manipkd_lagrange_lr 1.0
            --manipkd_project true
        )
    fi
    case "$defence" in
        Krum) defence_args=(--krum_f 9 --krum_num_selected 1) ;;
        TriMean) defence_args=(--trimean_trim_ratio 0.3) ;;
    esac
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
            --probe_core_sigma_threshold 7.0
        )
    fi

    {
        echo "Start: $(date)"
        echo "Run ID: $run_id; attack: $attack; defence: $defence; SPC: $use_spc"
        echo "Python: $PYTHON"
        echo "Checkpoint: $CHECKPOINT"
    } > "$log"

    if "$PYTHON" -u main.py \
        --debug \
        --run_id "$run_id" \
        --seed 20260805 \
        --dataset cifar10 \
        --arch ResNet18 \
        --batch_size 128 \
        --strategy FedDistill \
        --attack "$attack" \
        --defence "$defence" \
        --memory_method none \
        --n_clients 30 \
        --n_byzantine_clients "$n_byzantine" \
        --n_total_local_epochs 70 \
        --n_communications 70 \
        --n_server_epochs_per_round 5 \
        --server_lr "(Linear, 0.01)" \
        --client_lr "(Linear, 0.1)" \
        --restart_client_lr false \
        --reinit_server false \
        --server_init_ckpt "$CHECKPOINT" \
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
        "${defence_args[@]}" \
        "${probe_args[@]}" \
        "${attack_args[@]}" \
        >> "$log" 2>&1; then
        status=0
    else
        status=$?
    fi

    {
        echo "Exit code: $status"
        echo "End: $(date)"
    } >> "$log"
    {
        echo "Run $run_id: attack=$attack, defence=$defence, SPC=$use_spc, exit=$status"
        grep -F 'Final accuracy summary' "$log" | tail -n 1 || true
        grep -F '[ManipulatingKD diagnostics]' "$log" | tail -n 1 || true
        grep -F '[Probe Core-Sigma Filter]' "$log" | tail -n 1 || true
        echo
    } | tee -a "$SUMMARY"
    return "$status"
}

for ((i = START_INDEX; i < ${#EXPERIMENT_NAMES[@]}; i++)); do
    use_spc=false
    if [ "$i" -ge 8 ]; then
        use_spc=true
    fi
    run_exp "$((BASE_ID + i - START_INDEX))" \
        "${EXPERIMENT_NAMES[$i]}" "${ATTACKS[$i]}" "${DEFENCES[$i]}" "$use_spc"
done

echo "All scheduled experiments finished: $(date)" | tee -a "$SUMMARY"
