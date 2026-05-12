#!/usr/bin/env bash


set -euo pipefail
cd /workspace/Pothole-I

echo "═══════════════════════════════════════════════════════════════"
echo "  AI Eyes on the Road v2 — RunPod Setup & Start"
echo "═══════════════════════════════════════════════════════════════"

#Directories 
mkdir -p uploads outputs static/snapshots database models runs logs .cache/torch .cache/huggingface

#Env vars─
if [ -f .env.gpu ]; then
    set -a; source .env.gpu; set +a
    echo "✓ Loaded .env.gpu"
else
    echo "✗ WARNING: .env.gpu not found"
fi

#Model auto-restore
echo ""
echo "▶ Checking for trained pothole model..."
if [ -f "models/pothole_yolov8.pt" ]; then
    echo "✓ Trained model found: models/pothole_yolov8.pt"
elif [ -n "${HF_TOKEN:-}" ] && [ -n "${HF_MODEL_REPO:-}" ]; then
    echo "  No local model found — restoring latest weights from HuggingFace Hub..."
    #Ensure huggingface_hub is available before trying to pull
    pip install --quiet "huggingface_hub>=0.20.0" 2>/dev/null || true
    if python3 training/model_store.py pull --dest models/; then
        echo "✓ Model restored from Hub."
    else
        echo "⚠ Hub restore failed — app will start with base YOLO weights."
        echo "  Check HF_TOKEN / HF_MODEL_REPO in .env.gpu, then restart."
    fi
else
    echo "⚠ No trained model found and HF_TOKEN/HF_MODEL_REPO not set."
    echo "  App will start with base YOLO weights. To fix:"
    echo "    1. Train:   python training/train_gpu.py"
    echo "    2. Add HF_TOKEN and HF_MODEL_REPO to .env.gpu for automatic restore on next pod."
fi

#Python dependencies──────
echo ""
echo "▶ Installing Python dependencies..."
pip install blinker --ignore-installed --quiet 2>/dev/null || true
#Skip torch/torchvision — RunPod PyTorch images have the correct CUDA wheels pre-installed
grep -vE "^(torch|torchvision|torchaudio|onnxruntime)" requirements.txt | \
    grep -v "^#" | grep -v "^[[:space:]]*$" | \
    pip install -r /dev/stdin --quiet
echo "✓ Dependencies installed."

#GPU check
echo ""
echo "▶ GPU check..."
python3 -c "
import torch
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    total = torch.cuda.get_device_properties(0).total_memory
    print('VRAM:', round(total / 1e9, 1), 'GB')
"

#Redis
echo ""
echo "▶ Starting Redis..."
pkill -f "redis-server" 2>/dev/null || true
sleep 1

#Install redis-server if missing (happens on fresh PyTorch pods)
if ! command -v redis-server &>/dev/null; then
    echo "  Installing redis-server..."
    #Kill any stale apt/dpkg processes from prior failed runs (they stay as T-state zombies
    #holding the dpkg lock, causing every subsequent apt-get to hang indefinitely)
    pkill -9 -f "apt-get" 2>/dev/null || true
    pkill -9 -f "dpkg"    2>/dev/null || true
    sleep 2
    #Remove stale lock files — safe because we just killed all holders above
    rm -f /var/lib/dpkg/lock-frontend \
          /var/lib/dpkg/lock \
          /var/cache/apt/archives/lock \
          /var/lib/apt/lists/lock
    dpkg --configure -a -q 2>/dev/null || true
    apt-get update -qq && apt-get install -y redis-server -qq
fi

#Start Redis on default port, no persistence, bind localhost only
redis-server --daemonize yes \
    --logfile logs/redis.log \
    --save "" \
    --appendonly no \
    --bind 127.0.0.1 \
    --port 6379

sleep 1
if redis-cli ping | grep -q PONG; then
    echo "✓ Redis is up."
else
    echo "✗ Redis failed to start. Check logs/redis.log"
    exit 1
fi

#Override CELERY_BROKER_URL to use localhost (not 'redis' container hostname)
export CELERY_BROKER_URL=redis://127.0.0.1:6379/0
export CELERY_RESULT_BACKEND=redis://127.0.0.1:6379/1

#Celery worker────────────
echo ""
echo "▶ Starting Celery worker..."
pkill -f "celery.*worker" 2>/dev/null || true
sleep 1

nohup celery -A services.video_queue:get_celery_app worker \
    --loglevel=info \
    --concurrency=1 \
    --pool=solo \
    -Q celery \
    > logs/celery.log 2>&1 &
CELERY_PID=$!
echo "✓ Celery worker started (PID $CELERY_PID)"

#Flask app
echo ""
echo "▶ Starting Flask app..."
pkill -f "python3 app.py" 2>/dev/null || true
pkill -f "python app.py"  2>/dev/null || true
sleep 1

nohup python3 app.py > logs/app.log 2>&1 &
APP_PID=$!
echo "✓ Flask app started (PID $APP_PID)"

sleep 5
if kill -0 $APP_PID 2>/dev/null; then
    echo "✓ App is running."
    echo ""
    echo "── Last log lines"
    tail -20 logs/app.log
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  Services running:"
    echo "    Redis       → redis://127.0.0.1:6379"
    echo "    Celery      → PID $CELERY_PID  (logs/celery.log)"
    echo "    Flask app   → http://0.0.0.0:5001  (logs/app.log)"
    echo ""
    echo "  Access the dashboard:"
    echo "    RunPod HTTP proxy: https://<pod-id>-5001.proxy.runpod.net"
    echo "    SSH tunnel:  ssh -L 5001:localhost:5001 root@<host> -p <port>"
    echo "    Then open:   http://localhost:5001"
    echo ""
    echo "  Tail logs:"
    echo "    tail -f logs/app.log"
    echo "    tail -f logs/celery.log"
    echo "═══════════════════════════════════════════════════════════════"
else
    echo "✗ App failed to start. Last 30 lines of log:"
    tail -30 logs/app.log
    exit 1
fi
