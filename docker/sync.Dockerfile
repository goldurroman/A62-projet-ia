FROM python:3.10-slim

# Dossier de travail
WORKDIR /app

# Dépendances système nécessaires pour DVC + Git + MLflow
RUN apt-get update && apt-get install -y \
    git \
    curl \
    gcc \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*



# Installer DVC + S3 + YOLOv8 + MLflow
RUN pip install --no-cache-dir \
    dvc \
    dvc-s3 \
    mlflow \
    ultralytics

COPY docker/debug.sh /app/debug.sh
RUN chmod +x /app/debug.sh


# Copier le projet dans l'image
COPY .dvc /app/.dvc
COPY docker/sync.py /app/sync.py
COPY src /app/src
COPY data /app/data
COPY data.dvc /app/data.dvc
COPY train.py /app/train.py

# Variables d'environnement pour MinIO dans Kubernetes
ENV AWS_ACCESS_KEY_ID=minioadmin
ENV AWS_SECRET_ACCESS_KEY=minioadmin

# On exécute debug.sh, puis sync.py
CMD ["/bin/sh", "-c", "/app/debug.sh && python /app/sync.py"]