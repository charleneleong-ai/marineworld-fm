#!/usr/bin/env bash
# queue_v2_gpu.sh — Watch A100 GPU, auto-launch v2 pretraining when free.
#
# Usage (from local machine):
#   bash scripts/queue_v2_gpu.sh
#
# Or run detached:
#   setsid nohup bash scripts/queue_v2_gpu.sh >>logs/queue_v2_$(date -u +%Y%m%dT%H%M%SZ).log 2>&1 &
#
# The script:
#   1. Polls the A100 every 60s until GPU util < 10% and mem < 20GB.
#   2. Syncs code (git pull) on the remote.
#   3. Launches the v2 training run in a detached tmux session.
#   4. Exits.

set -euo pipefail

REMOTE="pi-a100-80gb"
REMOTE_DIR="/home/ubuntu/marineworld-fm"
FVESSEL_ROOT="${REMOTE_DIR}/data/raw/fvessel/Clip-10"
POLL_INTERVAL=60
UTIL_THRESHOLD=10    # GPU util % below which GPU is "free"
MEM_THRESHOLD=20000  # Memory used (MiB) below which GPU is "free"
RUN_NAME="v2_crossmodal_$(date -u +%Y%m%dT%H%M%SZ)"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

wait_for_gpu() {
    log "Polling ${REMOTE} for free GPU..."
    while true; do
        info=$(ssh "$REMOTE" nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null || echo "99,99999")
        util=$(echo "$info" | cut -d',' -f1 | tr -d ' ')
        mem=$(echo "$info" | cut -d',' -f2 | tr -d ' ')
        log "GPU util=${util}% mem=${mem}MiB"
        if [ "$util" -lt "$UTIL_THRESHOLD" ] && [ "$mem" -lt "$MEM_THRESHOLD" ]; then
            log "GPU free! Launching v2 run."
            return 0
        fi
        sleep "$POLL_INTERVAL"
    done
}

sync_code() {
    log "Syncing code on ${REMOTE}..."
    ssh "$REMOTE" bash -c "'
        cd ${REMOTE_DIR} &&
        git fetch origin &&
        git reset --hard origin/main &&
        uv sync --all-extras 2>&1 | tail -3
    '"
}

launch_run() {
    log "Launching v2 pretraining as tmux session: ${RUN_NAME}"
    ssh "$REMOTE" bash -c "'
        cd ${REMOTE_DIR} &&
        tmux new-session -d -s ${RUN_NAME} \
            \"WANDB_MODE=online uv run --extra train python -m marineworld.train.pretrain \
                --config-name config_v2 \
                data.root=${FVESSEL_ROOT} \
                runtime=a100 \
                train.epochs=100 \
                2>&1 | tee outputs/${RUN_NAME}.log\"
    '"
    log "Run launched. Monitor with: ssh ${REMOTE} tmux attach -t ${RUN_NAME}"
    log "Or check W&B: https://wandb.ai/chaleong/marineworld-fm"
}

# --- main ---
log "Starting v2 GPU queue watcher."
wait_for_gpu
sync_code
launch_run
log "Done. Run is queued on ${REMOTE}."
