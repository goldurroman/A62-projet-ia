# gen_container_name.py
from pathlib import Path

TRAIN_IMAGES = Path("data/train/images")
VAL_IMAGES = Path("data/val/images")

def list_images(path: Path):
    return sorted([p for p in path.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])

def read_config_from_train_py():
    cfg = {
        "MODEL_NAME": None,
        "IMG_SIZE": None,
        "COLOR_MODE": None,
    }

    with open("train.py", "r", encoding="utf-8") as f:
        for raw_line in f:
            # enlever les commentaires
            line = raw_line.split("#", 1)[0].strip()

            if not line:
                continue

            # MODEL_NAME
            if cfg["MODEL_NAME"] is None and line.startswith("MODEL_NAME"):
                if "=" in line:
                    value = line.split("=", 1)[1].strip()
                    value = value.replace('"', '').replace("'", "")
                    cfg["MODEL_NAME"] = value
                continue

            # IMG_SIZE
            if cfg["IMG_SIZE"] is None and line.startswith("IMG_SIZE"):
                if "=" in line:
                    value = line.split("=", 1)[1].strip()
                    cfg["IMG_SIZE"] = value
                continue

            # COLOR_MODE
            if cfg["COLOR_MODE"] is None and line.startswith("COLOR_MODE"):
                if "=" in line:
                    value = line.split("=", 1)[1].strip()
                    value = value.replace('"', '').replace("'", "")
                    cfg["COLOR_MODE"] = value
                continue

            # stop si tout est trouvé
            if all(cfg.values()):
                break

    return cfg

def generate_name(cfg):
    base_model = cfg["MODEL_NAME"].replace("-seg.pt", "").replace(".pt", "")
    train_count = len(list_images(TRAIN_IMAGES))
    val_count = len(list_images(VAL_IMAGES))
    return f"{base_model}-{train_count}-{val_count}-{cfg['IMG_SIZE']}-{cfg['COLOR_MODE']}"

if __name__ == "__main__":
    cfg = read_config_from_train_py()
    name = generate_name(cfg)
    print(name)
