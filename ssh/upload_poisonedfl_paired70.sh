#!/usr/bin/env bash
set -Eeuo pipefail

# Upload only the files required by the paired70 experiment.
# Usage:
#   SERVER=user@host REMOTE_DIR=/data/fdd-main bash ssh/upload_poisonedfl_paired70.sh
# Optional: SSH_PORT=22 IDENTITY_FILE=~/.ssh/id_ed25519
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
SERVER="${SERVER:-}"
REMOTE_DIR="${REMOTE_DIR:-}"
SSH_PORT="${SSH_PORT:-22}"
IDENTITY_FILE="${IDENTITY_FILE:-}"

if [[ -z "$SERVER" || -z "$REMOTE_DIR" ]]; then
    echo "Usage: SERVER=user@host REMOTE_DIR=/data/fdd-main bash $0" >&2
    exit 2
fi

SSH_ARGS=(-p "$SSH_PORT")
RSYNC_SSH="ssh -p $SSH_PORT"
if [[ -n "$IDENTITY_FILE" ]]; then
    SSH_ARGS+=(-i "$IDENTITY_FILE")
    RSYNC_SSH="ssh -p $SSH_PORT -i $IDENTITY_FILE"
fi

FILES=(
    main.py
    runners/PoisonedFLPairRunner.py
    runners/ScratchRunner.py
    byzantine/attacks.py
    byzantine/poisonedfl.py
    probes/probe_auditor.py
    tests/test_poisonedfl_pair.py
    tools/summarize_poisonedfl_pair.py
    ssh/run_poisonedfl_paired70.sh
    docs/poisonedfl_paired70.md
)

cd "$PROJECT_DIR"
for file in "${FILES[@]}"; do
    [[ -f "$file" ]] || { echo "Missing local file: $file" >&2; exit 1; }
done

echo "Creating remote directory: $SERVER:$REMOTE_DIR"
ssh "${SSH_ARGS[@]}" "$SERVER" "mkdir -p '$REMOTE_DIR'"

if command -v rsync >/dev/null 2>&1; then
    echo "Uploading with rsync"
    rsync -az --relative -e "$RSYNC_SSH" "${FILES[@]}" "$SERVER:$REMOTE_DIR/"
else
    echo "rsync not found; uploading with scp"
    for file in "${FILES[@]}"; do
        parent="$(dirname -- "$file")"
        ssh "${SSH_ARGS[@]}" "$SERVER" "mkdir -p '$REMOTE_DIR/$parent'"
        scp "${SSH_ARGS[@]}" "$file" "$SERVER:$REMOTE_DIR/$file"
    done
fi

echo "Verifying remote files"
ssh "${SSH_ARGS[@]}" "$SERVER" "cd '$REMOTE_DIR' && sha256sum main.py runners/PoisonedFLPairRunner.py tools/summarize_poisonedfl_pair.py ssh/run_poisonedfl_paired70.sh"
echo "Upload complete: $SERVER:$REMOTE_DIR"
