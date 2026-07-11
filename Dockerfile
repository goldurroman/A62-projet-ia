# ---------------------------------------------------------
# Base image : Python 3.10 slim (léger, stable, compatible YOLO)
# ---------------------------------------------------------
FROM python:3.10-slim

# ---------------------------------------------------------
# Dossier de travail
# ---------------------------------------------------------
WORKDIR /app

# ---------------------------------------------------------
# Dépendances système nécessaires pour OpenCV, Torch, Ultralytics
# ---------------------------------------------------------
RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------
# Installation des dépendances Python (MLflow, YOLO, etc.)
# ---------------------------------------------------------
COPY requirements-mlflow.txt /app/
RUN pip install --no-cache-dir -r requirements-mlflow.txt
# Installer explicitement DVC avec l'extension s3 (nécessaire pour MinIO)
RUN pip install --no-cache-dir "dvc[s3]"
# Copier explicitement le dossier de configuration DVC à l'intérieur de l'image
COPY .dvc /app/.dvc

RUN git init /app && \
    git config --global user.email "student@college.ca" && \
    git config --global user.name "Student"

# ---------------------------------------------------------
# Copie des scripts principaux (train.py, gen_container_name.py)
# ---------------------------------------------------------
COPY train.py /app/train.py
COPY gen_container_name.py /app/gen_container_name.py

# ---------------------------------------------------------
# Copie du code utilitaire (src/)
# ---------------------------------------------------------
COPY src /app/src

# ---------------------------------------------------------
# Copie des données (ISIC)
# ---------------------------------------------------------
#COPY data /app/data

# ---------------------------------------------------------
# Copie des fichiers de configuration
# ---------------------------------------------------------
COPY MLproject /app/MLproject
COPY Dockerfile /app/Dockerfile
COPY .dvcignore /app/.dvcignore
COPY .gitignore /app/.gitignore

# ---------------------------------------------------------
# MLflow tracking URI (utilisé par ton code, pas par YOLO)
# ---------------------------------------------------------
ENV MLFLOW_TRACKING_URI=sqlite:///mlflow.db

# ---------------------------------------------------------
# Commande d'exécution
# ---------------------------------------------------------
CMD ["python", "train.py"]
