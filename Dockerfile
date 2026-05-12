FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive


RUN rm -rf /var/lib/apt/lists/* \
    && for i in 1 2 3 4 5; do \
        apt-get -o Acquire::Retries=5 update \
        && apt-get -o Acquire::Retries=5 install -y --no-install-recommends \
            libglib2.0-0 \
            libgomp1 \
            libxcb1 \
            ffmpeg \
            curl \
        && break \
        || { echo "apt attempt $i failed, retrying in 30s..."; sleep 30; rm -rf /var/lib/apt/lists/*; }; \
    done \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu && \
    grep -vE "^(torch|torchvision|opencv-python)" requirements.txt | \
        pip install --no-cache-dir -r /dev/stdin && \
    pip uninstall -y opencv-python && \
    pip install --no-cache-dir "opencv-python-headless>=4.8.0"


COPY . .


RUN mkdir -p uploads outputs static/snapshots database models


ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOST=0.0.0.0 \
    PORT=5001 \
    DEBUG=False

EXPOSE 5001

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:${PORT:-5001}/api/health || exit 1

CMD ["python", "app.py"]
