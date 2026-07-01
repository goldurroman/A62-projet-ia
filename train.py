import os
import argparse
from pathlib import Path
from ultralytics import settings
print("[DEBUG- train.py] Avant update, settings.mlflow =", settings.get("mlflow"))
settings.update({"mlflow": False})
print("[DEBUG- train.py] Après update, settings.mlflow =", settings.get("mlflow"))
import re
import cv2
from PIL import Image
import mlflow
from ultralytics import YOLO

# =========================
# CONFIG (centralisé)
# =========================

MODEL_NAME = "yolov8n-seg.pt"   # yolov8n-seg.pt, yolov8s-seg.pt, yolov8m-seg.pt
IMG_SIZE = 384                  # 256, 512, 640, 768...
COLOR_MODE = "rgb"              # "rgb" ou "gray"
EPOCHS = 10                     # nombre d'époques
PATIENCE = 5                   # early stopping patience

DATA_DIR = Path("data")
TRAIN_IMAGES = DATA_DIR / "train" / "images"
TRAIN_LABELS = DATA_DIR / "train" / "labels"
VAL_IMAGES = DATA_DIR / "val" / "images"
VAL_LABELS = DATA_DIR / "val" / "labels"
DATA_YAML = DATA_DIR / "data.yaml"


# =========================
# UTILS
# =========================
def print_config():
    print(f"[INFO - train.py] MODEL_NAME={MODEL_NAME}")
    print(f"[INFO - train.py] IMG_SIZE={IMG_SIZE}")
    print(f"[INFO - train.py] COLOR_MODE={COLOR_MODE}")
    print(f"[INFO - train.py] TRAIN_IMAGES={len(list_images(TRAIN_IMAGES))}")
    print(f"[INFO - train.py] VAL_IMAGES={len(list_images(VAL_IMAGES))}")

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def list_images(path: Path):
    return sorted([p for p in path.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])


def create_yolo_label(mask_path: Path, label_path: Path, img_w: int, img_h: int):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"[WARN- train.py] Impossible de lire le masque : {mask_path}")
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


# =========================
# CONTAINER NAMING
# =========================

def generate_container_name(model_name: str, train_dir: Path, val_dir: Path, imgsz: int, color: str) -> str:
    train_count = len(list_images(train_dir))
    val_count = len(list_images(val_dir))

    base_model = Path(model_name).stem.replace("-seg", "").replace(".pt", "")

    return f"{base_model}-{train_count}-{val_count}-{imgsz}-{color}"


# =========================
# DATASET PREPARATION
# =========================

def prepare_local_isic_dataset():
    print("[INFO - train.py] Préparation du dataset ISIC")

    ensure_dir(TRAIN_LABELS)
    ensure_dir(VAL_LABELS)

    # TRAIN
    for img_path in list_images(TRAIN_IMAGES):
        mask_path = TRAIN_LABELS / f"{img_path.stem}_segmentation.png"
        if not mask_path.exists():
            print(f"[WARN- train.py] Masque manquant pour train: {img_path}")
            continue

        w, h = Image.open(img_path).size
        create_yolo_label(mask_path, TRAIN_LABELS / f"{img_path.stem}.txt", w, h)

    # VAL
    for img_path in list_images(VAL_IMAGES):
        mask_path = VAL_LABELS / f"{img_path.stem}_segmentation.png"
        if not mask_path.exists():
            print(f"[- train.py] Masque manquant pour val: {img_path}")
            continue

        w, h = Image.open(img_path).size
        create_yolo_label(mask_path, VAL_LABELS / f"{img_path.stem}.txt", w, h)

    # data.yaml
    data_yaml_content = f"""
train: /app/data/train/images
val: /app/data/val/images

names:
  0: lesion
""".strip()

    with open(DATA_YAML, "w") as f:
        f.write(data_yaml_content)


# =========================
# MAIN — YOLOv8 + MLflow PRO
# =========================

def main():
    print("[INFO - train.py] train.py running")

    # Renommage automatique du container
    container_name = generate_container_name(
        MODEL_NAME, TRAIN_IMAGES, VAL_IMAGES, IMG_SIZE, COLOR_MODE
    )

    # Préparation dataset
    prepare_local_isic_dataset()

    # Compter les images (pour MLflow)
    train_count = len(list_images(TRAIN_IMAGES))
    val_count = len(list_images(VAL_IMAGES))

    # Config MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("projet-synthese")

    print("[DEBUG- train.py] MLflow experiment actif :", mlflow.get_experiment_by_name("projet-synthese"))

    with mlflow.start_run(run_name=container_name):
        print("[DEBUG- train.py] MLflow run_id =", mlflow.active_run().info.run_id)
        # Log hyperparams
        mlflow.log_param("model_name", MODEL_NAME)
        mlflow.log_param("imgsz", IMG_SIZE)
        mlflow.log_param("color_mode", COLOR_MODE)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("patience", PATIENCE)

        # Log dataset size
        mlflow.log_param("train_images", train_count)
        mlflow.log_param("val_images", val_count)
        mlflow.log_param("container_name", container_name)

        print("[INFO - train.py] Chargement du modèle YOLO")
        model = YOLO(MODEL_NAME)

        results = model.train(
            data=DATA_YAML.as_posix(),
            epochs=EPOCHS,
            patience=PATIENCE,
            imgsz=IMG_SIZE,
            save=True,  # YOLO doit sauvegarder les artefacts
            exist_ok=True,  # éviter les erreurs si le dossier existe
            name=container_name,
            workers=0,
            deterministic=True,
            plots=True,
            project="models/runs",
        )
        # Log des artefacts YOLO dans MLflow
        mlflow.log_artifact(f"models/runs/{container_name}/weights/best.pt")
        mlflow.log_artifact(f"models/runs/{container_name}/weights/last.pt")
        # Images YOLO (labels, prédictions, courbes)
        mlflow.log_artifact(f"models/runs/{container_name}/labels.jpg")
        mlflow.log_artifact(f"models/runs/{container_name}/results.png")

        print("[INFO - train.py] Enregistrement des métriques dans MLflow")
        metrics = getattr(results, "results_dict", {}) or {}

        def clean_metric_name(name: str) -> str:
            name = re.sub(r"[()]", "", name)
            name = re.sub(r"[^a-zA-Z0-9_./ -]", "_", name)
            name = name.replace(" ", "_")
            return name

        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                clean_k = clean_metric_name(k)
                mlflow.log_metric(clean_k, float(v))

        run_dir = Path(results.save_dir)
        print("[INFO - train.py] Enregistrement des artefacts YOLO dans MLflow")
        for artifact in run_dir.glob("*"):
            if artifact.is_file():
                mlflow.log_artifact(str(artifact), artifact_path="yolo_artifacts")

        print("[INFO - train.py] Enregistrement des images YOLO dans MLflow")
        for artifact in run_dir.glob("*.jpg"):
            mlflow.log_artifact(str(artifact), artifact_path="yolo_images")

        for artifact in run_dir.glob("*.png"):
            mlflow.log_artifact(str(artifact), artifact_path="yolo_images")


        print("[INFO - train.py] Entraînement terminé, run MLflow enregistré.")


if __name__ == "__main__":
    main()
