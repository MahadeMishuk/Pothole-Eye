#!/usr/bin/env bash
#Deploy to RunPod via SSH/rsync
#Usage: bash scripts/deploy_runpod.sh
#
#SSH key authentication ONLY — no password prompts, ever.
#If SSH key auth fails the script aborts immediately.

set -euo pipefail

#RunPod connection─────────
#Update these from RunPod dashboard → My Pods → Connect → SSH over exposed TCP
RUNPOD_HOST="your.runpod.ip.here"
RUNPOD_PORT="your_port_here"
RUNPOD_USER="root"
SSH_KEY="$HOME/.ssh/id_ed25519"
REMOTE_DIR="/workspace/Pothole-I"



#Verify key file exists before attempting anything
if [[ ! -f "$SSH_KEY" ]]; then
    echo "✗ FATAL: SSH key not found at $SSH_KEY"
    echo "  Generate one with: ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519"
    exit 1
fi



#Strict SSH options — PasswordAuthentication=no ensures no fallback to password.
#IdentitiesOnly=yes forces ONLY the specified key, ignoring ssh-agent offers.
SSH_OPTS="-i $SSH_KEY \
  -p $RUNPOD_PORT \
  -o StrictHostKeyChecking=no \
  -o PasswordAuthentication=no \
  -o IdentitiesOnly=yes \
  -o ServerAliveInterval=60 \
  -o BatchMode=yes"

echo "═══════════════════════════════════════════════════════════"
echo "  AI Eyes on the Road — RunPod Sync"
echo "  Target: ${RUNPOD_USER}@${RUNPOD_HOST}:${RUNPOD_PORT}"
echo "  Remote path: ${REMOTE_DIR}"
echo "  SSH key: ${SSH_KEY}"
echo "═══════════════════════════════════════════════════════════"

#Step 1: Verify SSH key authentication before doing anything──
echo ""
echo "▶ Verifying SSH key authentication..."
if ssh $SSH_OPTS "${RUNPOD_USER}@${RUNPOD_HOST}" "echo SSH_OK" 2>/dev/null; then
    echo "✓ SSH key authentication succeeded."
else
    echo ""
    echo "✗ CRITICAL: SSH key authentication FAILED."
    echo ""
    echo "  Possible causes:"
    echo "    1. Key not authorised on RunPod pod — add your public key to the pod"
    echo "       via RunPod dashboard → Edit Pod → SSH Public Key."
    echo "    2. Wrong key path: $SSH_KEY"
    echo "    3. Pod IP/port changed — verify in RunPod dashboard."
    echo ""
    echo "  To diagnose:"
    echo "    ssh -vvv -i $SSH_KEY -p $RUNPOD_PORT ${RUNPOD_USER}@${RUNPOD_HOST}"
    echo ""
    echo "  Deployment aborted — fix SSH key auth first."
    exit 1
fi

#Step 2: Ensure remote workspace directory exists─
echo ""
echo "▶ Ensuring remote directory exists..."
ssh $SSH_OPTS "${RUNPOD_USER}@${RUNPOD_HOST}" "mkdir -p ${REMOTE_DIR}"
echo "✓ Remote directory ready."

#Step 3: Ensure rsync is available on remote
echo ""
echo "▶ Ensuring rsync is installed on remote..."
ssh $SSH_OPTS "${RUNPOD_USER}@${RUNPOD_HOST}" \
    "command -v rsync >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y rsync -qq; }"
echo "✓ rsync ready on remote."

#Step 4: Sync project files
echo ""
echo "▶ Syncing project files to RunPod..."

rsync -avz --no-owner --no-group --progress \
    --exclude='.git' \
    --exclude='.DS_Store' \
    --exclude='**/.DS_Store' \
    --exclude='__pycache__' \
    --exclude='**/__pycache__' \
    --exclude='*.pyc' \
    --exclude='.cache/' \
    --exclude='' \
    --exclude='uploads/*.mp4' \
    --exclude='uploads/*.avi' \
    --exclude='uploads/*.mov' \
    --exclude='outputs/' \
    --exclude='static/snapshots/' \
    --exclude='*.db' \
    --exclude='runs/' \
    -e "ssh -i $SSH_KEY \
          -p $RUNPOD_PORT \
          -o PasswordAuthentication=no \
          -o IdentitiesOnly=yes \
          -o StrictHostKeyChecking=no \
          -o BatchMode=yes" \
    "$(pwd)/" \
    "${RUNPOD_USER}@${RUNPOD_HOST}:${REMOTE_DIR}/"

#Step 5: Pull trained model weights back to Mac
echo ""
echo "▶ Checking for trained model weights on remote..."
if ssh $SSH_OPTS "${RUNPOD_USER}@${RUNPOD_HOST}" \
    "test -f ${REMOTE_DIR}/models/pothole_yolov8.pt" 2>/dev/null; then
    echo "  Found trained weights — syncing back to local models/ ..."
    mkdir -p "$(pwd)/models"
    rsync -avz --progress \
        -e "ssh -i $SSH_KEY \
              -p $RUNPOD_PORT \
              -o PasswordAuthentication=no \
              -o IdentitiesOnly=yes \
              -o StrictHostKeyChecking=no \
              -o BatchMode=yes" \
        "${RUNPOD_USER}@${RUNPOD_HOST}:${REMOTE_DIR}/models/" \
        "$(pwd)/models/"
    echo "✓ Model weights synced to local models/"
else
    echo "  No trained model on remote yet — train first, then re-run this script to pull weights back."
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  ✓ Sync complete!"
echo ""
echo "  OPTION A: Docker (recommended)─"
echo ""
echo "  1. SSH in:"
echo "     ssh -i $SSH_KEY -p ${RUNPOD_PORT} ${RUNPOD_USER}@${RUNPOD_HOST}"
echo ""
echo "  2. Rebuild + start (--build is REQUIRED after every code change):"
echo "     cd ${REMOTE_DIR}"
echo "     mkdir -p .cache/torch .cache/huggingface logs"
echo "     docker compose -f docker-compose.gpu.yml up --build -d"
echo ""
echo "  NOTE: 'docker compose up -d' (no --build) keeps the OLD image."
echo "  Always use --build after syncing code changes."
echo ""
echo "  3. After training, restart container to load new model weights:"
echo "     docker compose -f docker-compose.gpu.yml restart pothole-detector celery-worker"
echo ""
echo "  4. Verify GPU + model:"
echo "     docker exec ai-eyes-on-the-road-gpu python3 -c \\"
echo "       \"import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))\""
echo "     docker logs ai-eyes-on-the-road-gpu 2>&1 | grep -E 'model|Model|FOUND|pothole'"
echo ""
echo "  5. SSH tunnel for browser access:"
echo "     ssh -L 5001:localhost:5001 -i $SSH_KEY -p ${RUNPOD_PORT} ${RUNPOD_USER}@${RUNPOD_HOST}"
echo "     → http://localhost:5001"
echo ""
echo "  OPTION B: Bare-metal (start_app.sh)───"
echo ""
echo "  1. SSH in:"
echo "     ssh -i $SSH_KEY -p ${RUNPOD_PORT} ${RUNPOD_USER}@${RUNPOD_HOST}"
echo ""
echo "  2. Start services (Redis → Celery → Flask):"
echo "     cd ${REMOTE_DIR}"
echo "     bash scripts/start_app.sh"
echo ""
echo "  3. After training, restart to load new model:"
echo "     pkill -f 'python3 app.py' && pkill -f 'celery.*worker' && bash scripts/start_app.sh"
echo ""
echo "  NOTE: Bare-metal model path is auto-resolved from ${REMOTE_DIR}/models/"
echo "        (not the /app/models/ path in .env.gpu — that is Docker-only)."
echo "═══════════════════════════════════════════════════════════"
