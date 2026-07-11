FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

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
RUN pip install --no-cache-dir \
    mlflow==2.12.1 \
    boto3==1.34.84 \
    ultralytics==8.1.0 \
    opencv-python-headless==4.9.0.80 \
    pillow==10.2.0 \
    numpy==1.26.4 \
    pyyaml==6.0.1

# 3. Copie du script d'entraînement
COPY train.py /app/train.py

CMD ["python", "train.py"]