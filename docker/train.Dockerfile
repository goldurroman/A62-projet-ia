FROM python:3.10-slim

# Optimisation de Python et désactivation de la télémétrie Feast
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    FEAST_USAGE=False

# 1. Installation des dépendances système (librairies C/C++ requises par OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch 2.1.0 + Torchvision 0.16.0 (versions stables compatibles Ultralytics 8.1.0)
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cpu

# 2. Installation directe des packages Python (sans passer par un fichier txt)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    ultralytics==8.3.5 \
    opencv-python-headless==4.9.0.80 \
    pillow==10.2.0 \
    numpy==1.24.4 \
    pandas==1.5.3 \
    pyarrow==11.0.0 \
    dask==2023.3.1 \
    matplotlib==3.8.2 \
    boto3==1.34.84 \
    botocore==1.34.84 \
    requests==2.31.0 \
    feast==0.31.1 \
    redis==5.0.1 \
    s3fs==2024.3.1 \
    streamlit==1.32.0 \
    mlflow==2.12.1

# 3. Copie du script d'entraînement
COPY train.py /app/train.py

CMD ["python", "train.py"]