import os
import re
import sys
import logging
from pathlib import Path
import cv2
from PIL import Image
import mlflow
from ultralytics import YOLO, settings
import socket
import requests
import numpy as np
import boto3

os.environ["GIT_PYTHON_REFRESH"] = "quiet"

# ============================================================
# LOGGING PROFESSIONNEL (flush immédiat)
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s - %(asctime)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("train-worker")

log.info("=== [BOOT] Demarrage du script d'entrainement YOLOv8 ===")

# ============================================================
# ENV
# ============================================================
def get_env(name, default=None, cast=str):
    raw = os.getenv(name, default)
    try:
        return cast(raw)
    except Exception:
        log.error(f"[CONFIG] Impossible de caster {name}='{raw}' en {cast}")
        raise

MODEL_NAME = get_env("MODEL_NAME")
IMG_SIZE = get_env("IMG_SIZE", cast=int)
COLOR_MODE = get_env("COLOR_MODE")
EPOCHS = get_env("EPOCHS", cast=int)
PATIENCE = get_env("PATIENCE", cast=int)
RUN_NAME = get_env("RUN_NAME")

log.info(f"[CONFIG] MODEL_NAME={MODEL_NAME}")
log.info(f"[CONFIG] IMG_SIZE={IMG_SIZE}")
log.info(f"[CONFIG] COLOR_MODE={COLOR_MODE}")
log.info(f"[CONFIG] EPOCHS={EPOCHS}")
log.info(f"[CONFIG] PATIENCE={PATIENCE}")
log.info(f"[CONFIG] RUN_NAME={RUN_NAME}")

# ============================================================
# MLflow — Configuration initiale
# ============================================================
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-service:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "projet-synthese")

# Désactivation du plugin MLflow d'Ultralytics (évite les conflits)
settings.update({"mlflow": False})

# Application de l'URI de tracking
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
log.info(f"[MLFLOW] Tracking URI effective : {MLFLOW_TRACKING_URI}")

# ============================================================
# DIAGNOSTICS — DNS + HTTP avant mlflow.set_experiment()
# ============================================================

# Test DNS : résolution du service Kubernetes
try:
    resolved = socket.gethostbyname("mlflow-service")
    log.info(f"[DNS] mlflow-service resolved to {resolved}")
except Exception as e:
    log.error(f"[DNS] mlflow-service resolution failed: {e}")

# Test HTTP : vérification de l'accessibilité du serveur MLflow
try:
    test_url = f"{MLFLOW_TRACKING_URI}/api/2.0/mlflow/experiments/list"
    r = requests.get(test_url)
    log.info(f"[HTTP] MLflow reachable at {test_url}, status={r.status_code}")
except Exception as e:
    log.error(f"[HTTP] MLflow unreachable at {test_url}: {e}")

# ============================================================
# MLflow — Sélection de l'expérience
# ============================================================
try:
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    log.info(f"[MLFLOW] Expérience sélectionnée : {MLFLOW_EXPERIMENT_NAME}")
except Exception as e:
    log.error(f"[MLFLOW] Impossible de sélectionner l'expérience : {e}")
    raise

# ============================================================
# CHEMINS OPTIMISÉS POUR YOLO
# ============================================================
DATA_DIR = Path("/app/data")          # PVC (source)
WORKDIR = Path("/app/workdir")        # emptyDir (RAM)

LOCAL_DATA = WORKDIR / "dataset"            # dataset YOLO complet en RAM

TRAIN_IMAGES = LOCAL_DATA / "train" / "images"
TRAIN_LABELS = LOCAL_DATA / "train" / "labels"
VAL_IMAGES = LOCAL_DATA / "val" / "images"
VAL_LABELS = LOCAL_DATA / "val" / "labels"

DATA_YAML = WORKDIR / "data.yaml"

log.info(f"[DATA] DATA_DIR = {DATA_DIR}")
log.info(f"[DATA] WORKDIR = {WORKDIR}")
log.info(f"[DATA] LOCAL_DATA = {LOCAL_DATA}")

# ============================================================
# UTILITAIRES
# ============================================================
def list_images(path: Path):
    if not path.exists():
        log.warning(f"[DATA] Dossier inexistant : {path}")
        return []
    imgs = sorted([p for p in path.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    log.info(f"[DATA] {len(imgs)} images trouvees dans {path}")
    return imgs

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    log.info(f"[FS] Dossier cree/present : {path}")

def safe_symlink(src: Path, dst: Path):
    """Création idempotente de symlink (ne crash jamais)."""
    try:
        relative_src = os.path.relpath(src, dst.parent)
        if not dst.exists():
            os.symlink(relative_src, dst)
        elif dst.is_symlink():
            if os.readlink(dst) != relative_src:
                dst.unlink()
                os.symlink(relative_src, dst)
    except Exception as e:
        log.warning(f"[FS] Impossible de creer le symlink {dst}: {e}")

def create_yolo_label(mask_path: Path, label_path: Path, img_w: int, img_h: int):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        log.warning(f"[LABEL] Masque introuvable : {mask_path}")
        return
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    with open(label_path, "w") as f:
        for c in contours:
            if c.shape[0] > 2:
                pts = c.reshape(-1, 2)
                norm = [(p[0] / img_w, p[1] / img_h) for p in pts]
                line = "0 " + " ".join(f"{v:.6f}" for p in norm for v in p)
                f.write(line + "\n")
    log.info(f"[LABEL] Fichier YOLO genere : {label_path}")

# ============================================================
# PRÉPARATION DU DATASET EN RAM
# ============================================================
def prepare_dataset():
    # Validation du cache persistant
    if (TRAIN_IMAGES.exists() and any(TRAIN_IMAGES.iterdir()) and
            TRAIN_LABELS.exists() and any(TRAIN_LABELS.iterdir())):
        log.info(
            "[DATACACHE] Dataset et labels YOLO déjà présents sur le stockage persistant Linux. Skip de la préparation.")
        return

    log.info("[DATA] Preparation de la structure d'entrainement...")

    ensure_dir(TRAIN_IMAGES)
    ensure_dir(TRAIN_LABELS)
    ensure_dir(VAL_IMAGES)
    ensure_dir(VAL_LABELS)

    # -----------------------------
    # TRAIN
    # -----------------------------
    src_train_images = DATA_DIR / "train" / "images"
    src_train_labels = DATA_DIR / "train" / "labels"

    for img_path in list_images(src_train_images):
        local_img = TRAIN_IMAGES / img_path.name

        # Gestion du mode couleur
        if COLOR_MODE.lower() == "gray":
            img = Image.open(img_path).convert("L")
            img.save(local_img)
        else:
            safe_symlink(img_path, local_img)

        mask_path = src_train_labels / f"{img_path.stem}_segmentation.png"
        if mask_path.exists():
            w, h = Image.open(img_path).size
            create_yolo_label(mask_path, TRAIN_LABELS / f"{img_path.stem}.txt", w, h)
        else:
            log.warning(f"[DATA] Masque manquant pour {img_path.name} (train)")

    # -----------------------------
    # VAL
    # -----------------------------
    src_val_images = DATA_DIR / "val" / "images"
    src_val_labels = DATA_DIR / "val" / "labels"

    for img_path in list_images(src_val_images):
        local_img = VAL_IMAGES / img_path.name

        if COLOR_MODE.lower() == "gray":
            img = Image.open(img_path).convert("L")
            img.save(local_img)
        else:
            safe_symlink(img_path, local_img)

        mask_path = src_val_labels / f"{img_path.stem}_segmentation.png"
        if mask_path.exists():
            w, h = Image.open(img_path).size
            create_yolo_label(mask_path, VAL_LABELS / f"{img_path.stem}.txt", w, h)
        else:
            log.warning(f"[DATA] Masque manquant pour {img_path.name} (val)")

    # -----------------------------
    # data.yaml
    # -----------------------------
    data_yaml_content = (
        f"path: {LOCAL_DATA.resolve().as_posix()}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        "names:\n  0: lesion\n"
    )

    with open(DATA_YAML, "w") as f:
        f.write(data_yaml_content)

    log.info(f"[DATA] data.yaml genere : {DATA_YAML}")


# ============================================================================
# BLOC FEAST : INGESTION NON BLOQUANTE DES FEATURES MORPHOLOGIQUES (TRAIN JOB)
# ============================================================================

import os
import uuid
import boto3
import pandas as pd
from datetime import datetime
import warnings
from urllib3.exceptions import InsecureRequestWarning

def ingest_dataset_features_to_feast():
    """
    Ingestion des features morphologiques du dataset d'entraînement dans Feast.
    Version adaptée pour le job Kubernetes (train.py), avec les mêmes étapes
    que Streamlit : patch boto3, écriture S3 manuelle, écriture Redis via Feast.
    """

    try:
        log.info("[FEAST] Initialisation de l'ingestion des features du dataset...")

        # ----------------------------------------------------------------------
        # 1. Désactivation télémétrie Feast (déjà gérée dans Dockerfile)
        # ----------------------------------------------------------------------
        os.environ["FEAST_USAGE"] = "False"

        # ----------------------------------------------------------------------
        # 2. Suppression des warnings SSL (comme dans Streamlit)
        # ----------------------------------------------------------------------
        warnings.filterwarnings("ignore", category=InsecureRequestWarning)
        warnings.filterwarnings("ignore", message="Certificate did not match expected hostname")

        # ----------------------------------------------------------------------
        # 3. Configuration MinIO (identique à Streamlit)
        # ----------------------------------------------------------------------
        minio_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        minio_secret = os.getenv("MINIO_SECRET_KEY", "minioadmin")
        minio_endpoint = "http://minio-service:9000"

        os.environ["AWS_ACCESS_KEY_ID"] = minio_key
        os.environ["AWS_SECRET_ACCESS_KEY"] = minio_secret
        os.environ["FEAST_S3_ENDPOINT_URL"] = minio_endpoint

        # Patch boto3 pour forcer MinIO (comme Streamlit)
        boto3.setup_default_session(
            aws_access_key_id=minio_key,
            aws_secret_access_key=minio_secret,
        )

        # ----------------------------------------------------------------------
        # 4. RepoConfig dynamique (identique à Streamlit)
        # ----------------------------------------------------------------------
        from feast import FeatureStore
        from feast.repo_config import RepoConfig
        from feast.infra.online_stores.redis import RedisOnlineStoreConfig

        config = RepoConfig(
            project="a62_project_synthese",
            provider="local",
            registry="s3://feast-registry/registry.db",
            online_store=RedisOnlineStoreConfig(
                connection_string="redis-service.default.svc.cluster.local:6379"
            ),
            s3_endpoint_url=minio_endpoint,
            entity_key_serialization_version=2
        )

        store = FeatureStore(config=config)
        log.info("[FEAST] FeatureStore initialisé avec succès.")

        # ----------------------------------------------------------------------
        # 5. Construction du dataset morphologique (identique à Streamlit)
        # ----------------------------------------------------------------------
        def calculate_morphology_for_feast(mask_path: Path) -> dict:
            """
            Calcule les métriques de morphologie à partir d'un masque de segmentation.
            Version identique à Streamlit.
            """
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                return None

            # Binarisation
            _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return None

            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)

            if area == 0 or perimeter == 0:
                return None

            # 1. Compacité
            compactness = (4 * np.pi * area) / (perimeter ** 2)

            # 2. Diamètre équivalent
            diameter_px = 2 * np.sqrt(area / np.pi)

            # 3. Asymétrie
            x, y, w, h = cv2.boundingRect(c)
            roi = binary[y:y + h, x:x + w]
            h_roi, w_roi = roi.shape
            top = float(np.sum(roi[0:h_roi // 2, :]))
            bottom = float(np.sum(roi[h_roi // 2:, :]))
            asymmetry = abs(top - bottom) / (top + bottom + 1e-6)

            # 4. Score de suspicion
            suspicion_score = 100.0 if asymmetry > 0.5 else 30.0

            return {
                "compactness": float(compactness),
                "asymmetry": float(asymmetry),
                "diameter_px": float(diameter_px),
                "suspicion_score": float(suspicion_score)
            }

        records = []

        src_train_images = DATA_DIR / "train" / "images"
        src_train_labels = DATA_DIR / "train" / "labels"

        for img_path in src_train_images.glob("*"):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue

            mask_path = src_train_labels / f"{img_path.stem}_segmentation.png"
            if not mask_path.exists():
                continue

            metrics = calculate_morphology_for_feast(mask_path)
            if not metrics:
                continue

            lesion_id = str(hash(img_path.stem) & 0xffffffff)

            records.append({
                "lesion_id": lesion_id,
                "event_timestamp": datetime.utcnow(),
                "compactness": metrics["compactness"],
                "asymmetry": metrics["asymmetry"],
                "diameter_px": metrics["diameter_px"],
                "suspicion_score": metrics["suspicion_score"]
            })

        if not records:
            log.warning("[FEAST] Aucun prédicteur calculé.")
            return

        df = pd.DataFrame(records)
        df["lesion_id"] = df["lesion_id"].astype(str)
        df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])

        log.info(f"[FEAST] Ingestion de {len(df)} lignes de caractéristiques...")

        # ----------------------------------------------------------------------
        # 6. ÉCRITURE OFFLINE → MinIO (identique à Streamlit)
        # ----------------------------------------------------------------------
        try:
            log.info("[FEAST] Tentative écriture OFFLINE → s3://feast-offline/lesion_morphology.parquet")

            s3 = boto3.client(
                "s3",
                endpoint_url=minio_endpoint,
                aws_access_key_id=minio_key,
                aws_secret_access_key=minio_secret,
            )

            import io
            buffer = io.BytesIO()
            df.to_parquet(buffer, index=False)
            buffer.seek(0)

            s3.put_object(
                Bucket="feast-offline",
                Key="lesion_morphology.parquet",
                Body=buffer.getvalue(),
                ContentType="application/octet-stream",
            )

            log.info("[FEAST] Écriture OFFLINE S3 complétée.")

        except Exception as e:
            log.warning(f"[FEAST][OFFLINE-S3] ERREUR écriture offline S3: {e}")

        # ----------------------------------------------------------------------
        # 7. ÉCRITURE ONLINE → Redis (identique à Streamlit)
        # ----------------------------------------------------------------------
        try:
            log.info("[FEAST] Tentative écriture ONLINE (Redis)...")

            store.write_to_online_store(
                feature_view_name="lesion_morphology",
                df=df,
            )

            log.info("[FEAST] Écriture ONLINE Redis complétée.")

        except Exception as e:
            log.warning(f"[FEAST][ONLINE] ERREUR écriture online: {e}")

    except Exception as e:
        log.warning(f"[FEAST] [NON-BLOCANT] Échec global Feast : {e}")
# ============================================================
# MAIN
# ============================================================
def main():
    log.info(f"[MLFLOW] Tracking URI : {mlflow.get_tracking_uri()}")

    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        experiments = client.search_experiments()
        log.info(f"[MLFLOW] Connexion OK. Expériences : {[e.name for e in experiments]}")
    except Exception as e:
        log.error(f"[MLFLOW] ERREUR : Impossible de lister les expériences : {e}")

    try:
        log.info("[START] Initialisation du pipeline YOLOv8")

        prepare_dataset()
        # =======================================================================
        # APPEL FEAST (NON BLOQUANT)
        # Calcule et pousse les caractéristiques des lésions d'entraînement
        # =======================================================================
        ingest_dataset_features_to_feast()
        # =======================================================================

        train_count = len(list_images(TRAIN_IMAGES))
        val_count = len(list_images(VAL_IMAGES))
        log.info(f"[DATA] Train={train_count} | Val={val_count}")

        with mlflow.start_run(run_name=RUN_NAME):
            run_id = mlflow.active_run().info.run_id
            log.info(f"[MLFLOW] Run démarré : ID={run_id}, name={RUN_NAME}")

            mlflow.log_param("model_name", MODEL_NAME)
            mlflow.log_param("imgsz", IMG_SIZE)
            mlflow.log_param("color_mode", COLOR_MODE)
            mlflow.log_param("epochs", EPOCHS)
            mlflow.log_param("patience", PATIENCE)
            mlflow.log_param("train_images", train_count)
            mlflow.log_param("val_images", val_count)

            log.info("[TRAIN] Chargement du modèle YOLOv8...")
            model = YOLO(MODEL_NAME)

            log.info("[TRAIN] Début de l'entraînement...")
            results = model.train(
                data=DATA_YAML.as_posix(),
                epochs=EPOCHS,
                patience=PATIENCE,
                imgsz=IMG_SIZE,
                batch=8,
                save=True,
                exist_ok=True,
                name=RUN_NAME,
                workers=0,
                deterministic=True,
                plots=True,
                project="models/runs",
            )
            log.info("[TRAIN] Entraînement terminé.")

            metrics = getattr(results, "results_dict", {}) or {}
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    clean_k = re.sub(r"[^a-zA-Z0-9_./ -]", "_", k).replace(" ", "_")
                    mlflow.log_metric(clean_k, float(v))
                    log.info(f"[METRIC] {clean_k}={v}")

            run_dir = Path(results.save_dir)
            log.info(f"[ARTIFACT] Dossier YOLO : {run_dir}")

            if run_dir.exists():
                # MLflow upload
                for artifact in run_dir.glob("**/*"):
                    if artifact.is_file():
                        rel_parent = artifact.relative_to(run_dir).parent
                        target_dir = Path("yolo_artifacts") / rel_parent
                        artifact_path = target_dir.as_posix()
                        if artifact_path.endswith("/.") or artifact_path == ".":
                            artifact_path = "yolo_artifacts"
                        mlflow.log_artifact(str(artifact), artifact_path=artifact_path)

                log.info("[ARTIFACT] Artefacts envoyés à MLflow.")

                # MinIO upload
                log.info("[MINIO] Initialisation client MinIO...")
                s3 = boto3.client(
                    "s3",
                    endpoint_url="http://minio-service:9000",
                    aws_access_key_id="minioadmin",
                    aws_secret_access_key="minioadmin",
                )

                visual_bucket = "yolo-visuals"
                run_folder = RUN_NAME

                existing_buckets = [b["Name"] for b in s3.list_buckets()["Buckets"]]
                if visual_bucket not in existing_buckets:
                    s3.create_bucket(Bucket=visual_bucket)
                    log.info(f"[MINIO] Bucket créé : {visual_bucket}")
                else:
                    log.info(f"[MINIO] Bucket existant : {visual_bucket}")

                for artifact in run_dir.glob("**/*"):
                    artifact_path = str(artifact)
                    if os.path.isfile(artifact_path):
                        key = f"{run_folder}/{os.path.relpath(artifact_path, run_dir)}"
                        s3.upload_file(artifact_path, visual_bucket, key)
                        log.info(f"[MINIO] Upload OK : {key}")

                log.info("[MINIO] Artefacts envoyés dans le bucket visuel.")

            else:
                log.warning("[ARTIFACT] Aucun artefact trouvé.")

        log.info(f"[MLFLOW] Run enregistré : {run_id}")
        log.info("[END] Script terminé avec succès.")

    except Exception as err:
        log.exception(f"[ERROR] Échec du script : {err}")
        raise

if __name__ == "__main__":
    main()

