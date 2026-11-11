#!/usr/bin/env bash
set -Eeuo pipefail

# Start the uploaded experiment detached on the remote server.
# Usage:
#   SERVER=user@host REMOTE_DIR=/data/fdd-main bash ssh/start_poisonedfl_paired70_remote.sh 11121 screen
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
SERVER="${SERVER:-}"
REMOTE_DIR="${REMOTE_DIR:-}"
SSH_PORT="${SSH_PORT:-22}"
IDENTITY_FILE="${IDENTITY_FILE:-}"
BASE_ID="${1:-11121}"
PROFILE="${2:-screen}"
ARCH="${ARCH:-ResNet18}"

if [[ -z "$SERVER" || -z "$REMOTE_DIR" ]]; then
    echo "Usage: SERVER=user@host REMOTE_DIR=/data/fdd-main bash $0 [base_id] [screen|smoke]" >&2
    exit 2
fi
if [[ ! "$BASE_ID" =~ ^[1-9][0-9]{0,8}$ || "$PROFILE" != screen && "$PROFILE" != smoke ]]; then
    echo "Invalid base ID or profile" >&2
    exit 2
fi

SSH_ARGS=(-p "$SSH_PORT")
if [[ -n "$IDENTITY_FILE" ]]; then SSH_ARGS+=(-i "$IDENTITY_FILE"); fi

echo "Starting remote paired70: $SERVER:$REMOTE_DIR, base=$BASE_ID, profile=$PROFILE, arch=$ARCH"
ssh "${SSH_ARGS[@]}" "$SERVER" "cd '$REMOTE_DIR' && mkdir -p logs && chmod +x ssh/run_poisonedfl_paired70.sh && ARCH='$ARCH' nohup bash ssh/run_poisonedfl_paired70.sh '$BASE_ID' '$PROFILE' run > 'logs/${BASE_ID}.launcher.log' 2>&1 < /dev/null & echo \$!"
echo "Use ssh $SERVER 'tail -f $REMOTE_DIR/logs/${BASE_ID}.launcher.log' to monitor."
