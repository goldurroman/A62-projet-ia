FROM python:3.10-slim

# Dossier de travail dans le container
WORKDIR /feast/repo

# Copie des fichiers Feast depuis le dossier feast/ (chemin relatif)
COPY ../feast/feature_store.yaml ./feature_store.yaml
COPY ../feast/features.py ./features.py

# Installation de Feast + dépendances S3
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    pandas==1.5.3 \
    pyarrow==11.0.0 \
    feast==0.31.1 \
    redis==5.0.1 \
    s3fs==2024.3.1 \
    dask==2023.3.1 \
    boto3==1.34.84 \
    botocore==1.34.84

# Variables d'environnement pour MinIO
ENV AWS_ACCESS_KEY_ID=minioadmin
ENV AWS_SECRET_ACCESS_KEY=minioadmin
ENV FEAST_S3_ENDPOINT_URL=http://minio-service:9000

CMD ["bash"]
