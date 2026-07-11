# ============================================================
# 1. Base image
# ============================================================
FROM python:3.10.14-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# ============================================================
# 2. System dependencies
# ============================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ============================================================
# 3. Python dependencies (versioning strict + cache)
# ============================================================
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.1.0 torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cpu

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    streamlit==1.32.0 \
    ultralytics==8.3.5 \
    opencv-python-headless==4.9.0.80 \
    pillow==10.2.0 \
    numpy==1.26.4 \
    matplotlib==3.8.2 \
    boto3==1.34.84 \
    botocore==1.34.84 \
    requests==2.31.0

# ============================================================
# 4. Copy application code (sans modèle)
# ============================================================
COPY app/streamlit-app.py /app/streamlit-app.py
COPY app/models/yolov8m-seg-best.pt /app/models/yolov8m-seg-best.pt

# ============================================================
# 5. Expose Streamlit port
# ============================================================
EXPOSE 8501

# ============================================================
# 6. Command
# ============================================================
CMD ["streamlit", "run", "streamlit-app.py", "--server.port=8501", "--server.address=0.0.0.0"]
