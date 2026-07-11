import os
import io
import uuid
import datetime
import logging
import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from ultralytics import YOLO
import warnings

# ============================================================
# 0. FUSEAU HORAIRE LOCAL (Montréal) + LOGGING
# ============================================================
try:
    import zoneinfo
    LOCAL_TZ = zoneinfo.ZoneInfo("America/Montreal")
except Exception:
    from datetime import timezone, timedelta
    LOCAL_TZ = timezone(timedelta(hours=-4))

warnings.filterwarnings("ignore", category=RuntimeWarning, module="streamlit")

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s - %(asctime)s] %(message)s",
)
log = logging.getLogger("Streamlit-App")

# ============================================================
# 1. CONFIGURATION INTERFACE & CONSTANTES
# ============================================================
st.set_page_config(
    page_title=" Pipeline YOLOv8-seg ",
    layout="wide",
    initial_sidebar_state="expanded",
)

MASK_THRESHOLD = 0.20      # seuil de binarisation
KERNEL_SIZE = 3            # kernel morphologique
MAX_IMAGE_SIZE = 2048

COLLECT_DATA_BUCKET = "user-drift-inputs"
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT_URL", "http://minio-service:9000")

log.info("[BOOT] Streamlit app démarrée ")

# ============================================================
# 2. CHARGEMENT DU MODÈLE YOLOv8-seg
# ============================================================
@st.cache_resource
def load_yolo_model():
    model_path = os.getenv("MODEL_PATH", "/app/models/yolov8m-seg-best.pt")
    log.info(f"[MODEL] Chargement du modèle YOLO depuis : {model_path}")
    model = YOLO(model_path)
    log.info(f"[MODEL] Noms des classes : {model.names}")
    return model

try:
    model = load_yolo_model()
except Exception as e:
    st.error(f"Erreur de chargement du modèle : {e}")
    log.error(f"[MODEL] ERREUR chargement modèle: {e}", exc_info=True)
    st.stop()

# ============================================================
# 3. SIDEBAR DIAGNOSTICS
# ============================================================
st.sidebar.header("🔍 Diagnostics YOLOv8-seg")
st.sidebar.write(f"**Task:** {model.task}")
st.sidebar.write(f"**Classes:** {model.names}")

conf_threshold = st.sidebar.slider("Seuil de confiance YOLO", 0.0, 1.0, 0.25, 0.01)
imgsz = st.sidebar.selectbox("Taille d'inférence (YOLO)", [256, 320, 640, 1024], index=2)

# ============================================================
# 4. FONCTION DE CHARGEMENT D'IMAGE
# ============================================================
def load_image_any_format_from_upload(uploaded_file, max_size=2048):
    try:
        img = Image.open(uploaded_file)
    except Exception as e:
        raise ValueError(f"Impossible de charger l'image : {e}")

    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")

    if max(img.size) > max_size:
        scale = max_size / max(img.size)
        new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
        img = img.resize(new_size, Image.LANCZOS)

    img_rgb = np.array(img)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    log.info(f"[LOAD] Image chargée et prétraitée : taille finale = {img_rgb.shape[1]}x{img_rgb.shape[0]}")
    return img_bgr, img_rgb

# ============================================================
# 5. FONCTIONS ANALYSE MORPHOLOGIQUE
# ============================================================
def compute_morphology_scores(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    compactness = (4 * np.pi * area) / (perimeter ** 2 + 1e-6)

    H, W = mask.shape
    left = mask[:, :W // 2].sum()
    right = mask[:, W // 2:].sum()
    top = mask[:H // 2, :].sum()
    bottom = mask[H // 2:, :].sum()

    asym_lr = abs(left - right) / (left + right + 1e-6)
    asym_tb = abs(top - bottom) / (top + bottom + 1e-6)
    asymmetry = max(asym_lr, asym_tb)

    (_, _), radius = cv2.minEnclosingCircle(cnt)
    diameter = radius * 2

    log.info(f"[MORPHO] Compacité={compactness:.3f}, Asymétrie={asymmetry:.3f}, Diamètre={diameter:.1f}")
    return {
        "compactness": float(compactness),
        "asymmetry": float(asymmetry),
        "diameter": float(diameter)
    }, cnt


def compute_global_suspicion_score(scores):
    comp = scores["compactness"]
    asym = scores["asymmetry"]
    diam = scores["diameter"]

    comp_score = (1 - comp) * 100
    asym_score = asym * 100
    diam_score = min(diam / 150, 1) * 100

    final_score = (
        0.40 * comp_score +
        0.40 * asym_score +
        0.20 * diam_score
    )

    final_score = min(100, max(0, final_score))
    log.info(f"[SCORE] Score global de suspicion = {final_score:.2f}")
    return final_score

# ============================================================
# 6. DRIFT TRACKING
# ============================================================
def save_image_for_drift_tracking(img_rgb):
    try:
        import boto3
        s3 = boto3.client(
            "s3",
            endpoint_url=MINIO_ENDPOINT,
            aws_access_key_id=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
            aws_secret_access_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        )

        from botocore.exceptions import ClientError
        try:
            s3.create_bucket(Bucket=COLLECT_DATA_BUCKET)
        except ClientError as e:
            if e.response["Error"]["Code"] not in ["BucketAlreadyExists", "BucketAlreadyOwnedByYou"]:
                raise

        now_local = datetime.datetime.now(LOCAL_TZ)
        timestamp = now_local.strftime("%Y-%m-%d_%H:%M:%S")
        filename = f"{timestamp}_{uuid.uuid4().hex[:8]}.jpg"

        _, buffer = cv2.imencode(".jpg", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        binary_io = io.BytesIO(buffer.tobytes())

        s3.put_object(
            Bucket=COLLECT_DATA_BUCKET,
            Key=filename,
            Body=binary_io,
            ContentType="image/jpeg",
        )
        log.info(f"[MINIO] Image uploadée: {filename}")
    except Exception as e:
        log.error(f"[MINIO] ERREUR upload: {e}", exc_info=True)


# ============================================================
# 7. UI PRINCIPALE — PIPELINE ÉTAPE PAR ÉTAPE
# ============================================================
st.title(" Pipeline YOLOv8-seg — Streamlit")
st.markdown("---")

uploaded_file = st.file_uploader("📤 Charger une image (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        img_bgr, img_rgb = load_image_any_format_from_upload(uploaded_file, max_size=MAX_IMAGE_SIZE)
    except Exception as e:
        st.error(f"Erreur de chargement de l'image : {e}")
        log.error(f"[UPLOAD] ERREUR chargement image: {e}", exc_info=True)
        st.stop()

    col1 = st.columns(1)[0]
    with col1:
        st.subheader("Image originale (RGB)")
        st.image(img_rgb)

    if st.button("🚀 LANCER LE PIPELINE (YOLOv8-seg)"):
        save_image_for_drift_tracking(img_rgb)

        with st.spinner("Inférence YOLOv8-seg en cours..."):
            try:
                results = model(img_bgr, verbose=False)
                res = results[0]
            except Exception as e:
                st.error("Erreur lors de l'inférence YOLO.")
                log.error(f"[YOLO] ERREUR inférence: {e}", exc_info=True)
                st.stop()

        st.write(f"[OK] Inférence terminée.")
        st.write(f"[INFO] Boxes détectées : {len(res.boxes)}")

        if res.masks is None or len(res.masks.data) == 0:
            st.error("❌ Aucun masque détecté par le modèle.")
            st.stop()

        st.write(f"[OK] Masques détectés : {res.masks.data.shape[0]}")
        st.write("[OK] Cellule d'inférence complétée.")

        num_masks = res.masks.data.shape[0]
        masks = res.masks.data.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()

        log.info(f"[DEBUG] Nombre de masques détectés : {num_masks}")
        log.info(f"[DEBUG] Surface des masques : {[m.sum() for m in masks]}")
        log.info(f"[DEBUG] Scores YOLO : {scores.tolist()}")

        H, W = masks.shape[1], masks.shape[2]
        min_area = 0.001 * (H * W)

        candidate_indices = []
        for i in range(masks.shape[0]):
            area = masks[i].sum()
            if area >= min_area:
                candidate_indices.append(i)

        if not candidate_indices:
            lesion_detected = False
            best_mask = np.zeros((H, W), dtype=np.uint8)
            st.warning("Tous les masques sont trop petits + masque vide.")
        else:
            candidate_scores = [scores[i] for i in candidate_indices]
            best_idx_local = int(np.argmax(candidate_scores))
            best_index = candidate_indices[best_idx_local]
            best_mask = masks[best_index]
            lesion_detected = True

            st.write(f"[OK] Masque sélectionné : #{best_index}")
            st.write(f"[INFO] Score YOLO = {scores[best_index]:.3f}")
            st.write(f"[INFO] Surface brute = {best_mask.sum():.0f} pixels")
            st.write("[OK] Sélection du meilleur masque complétée.")

        binary_mask = (best_mask >= MASK_THRESHOLD).astype(np.uint8)
        kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

        st.write("[OK] Masque final généré.")
        st.write(f"[INFO] Surface finale : {int(binary_mask.sum())} pixels")

        binary_mask_resized = cv2.resize(
            binary_mask,
            (img_rgb.shape[1], img_rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        mask_color = np.zeros_like(img_rgb)
        mask_color[..., 0] = binary_mask_resized * 255
        overlay = cv2.addWeighted(img_rgb, 1.0, mask_color, 0.4, 0)

        # ============================================================
        # GENERATION DES GRAPHIQUES (Grille 3 colonnes symétriques)
        # ============================================================
        st.markdown("---")
        st.header("📊 Visualisations du Pipeline")

        FIGURE_SIZE = (4, 4)

        # Fig 1 : Masque
        fig_mask, ax_mask = plt.subplots(figsize=FIGURE_SIZE)
        ax_mask.imshow(binary_mask, cmap="gray")
        ax_mask.axis("off")

        if binary_mask.sum() == 0:
            st.warning("Aucune lésion détectée + analyse morphologique impossible.")
        else:
            scores_morpho, cnt = compute_morphology_scores(binary_mask)

            if scores_morpho is not None:
                # Fig 2 : Contours
                hull = cv2.convexHull(cnt)
                vis = np.zeros((*binary_mask.shape, 3), dtype=np.uint8)
                cv2.drawContours(vis, [cnt], -1, (255, 0, 0), 2)
                cv2.drawContours(vis, [hull], -1, (0, 255, 0), 2)

                fig_contours, ax_contours = plt.subplots(figsize=FIGURE_SIZE)
                ax_contours.imshow(vis)
                ax_contours.axis("off")

                # Fig 3 : Heatmap d'asymétrie
                Hm, Wm = binary_mask.shape
                mask_norm = binary_mask.astype(float)

                left = mask_norm[:, :Wm // 2]
                right = mask_norm[:, Wm // 2:]
                right_flipped = np.fliplr(right)
                asym_lr = np.abs(left - right_flipped)
                asym_lr = cv2.resize(asym_lr, (Wm, Hm))

                top = mask_norm[:Hm // 2, :]
                bottom = mask_norm[Hm // 2:, :]
                bottom_flipped = np.flipud(bottom)
                asym_tb = np.abs(top - bottom_flipped)
                asym_tb = cv2.resize(asym_tb, (Wm, Hm))

                heatmap = asym_lr + asym_tb

                fig_heatmap, ax_heatmap = plt.subplots(figsize=FIGURE_SIZE)
                ax_heatmap.imshow(heatmap, cmap="hot")
                ax_heatmap.axis("off")

                # ⭐ Affichage aligné en 3 colonnes avec descriptifs Colab initiaux
                g1, g2, g3 = st.columns(3)

                with g1:
                    st.write("✅ [INFO] Masque final généré.")
                    st.subheader("**Masque Final**")
                    st.pyplot(fig_mask, bbox_inches='tight', pad_inches=0)

                with g2:
                    st.write("✅ [INFO] Visualisation des bords irréguliers…")
                    st.subheader("**Contours (Réel/Convexe)**")
                    st.pyplot(fig_contours, bbox_inches='tight', pad_inches=0)

                with g3:
                    st.write("✅ [INFO] Visualisation de la heatmap d'asymétrie…")
                    st.subheader("**Heatmap d'Asymétrie**")
                    st.pyplot(fig_heatmap, bbox_inches='tight', pad_inches=0)

                plt.close(fig_mask)
                plt.close(fig_contours)
                plt.close(fig_heatmap)

                # ============================================================
                # METRIQUES ET RÉSULTATS MORPHOLOGIQUES
                # ============================================================
                st.markdown("---")
                st.header("📊 Analyses Géométriques Avancées")

                comp = scores_morpho["compactness"]
                asym = scores_morpho["asymmetry"]
                diam = scores_morpho["diameter"]

                st.write("=== Analyse morphologique ===")
                st.write(f"· **Compacité :** {comp:.3f}")
                st.write(f"· **Asymétrie :** {asym:.3f}")
                st.write(f"· **Diamètre (px) :** {diam:.1f}")

                suspicion_score = compute_global_suspicion_score(scores_morpho)
                st.write(f"\n☒ **Score global de suspicion :** {suspicion_score:.1f} / 100")

                st.markdown("---")
                st.subheader("🖼️ Overlay final : image + masque YOLOv8-seg")
                st.image(overlay)

                st.write("\n=== Interprétation (non médicale) === ")
                st.write("Ce score reflète uniquement la complexité morphologique de la lésion")
                st.write("(forme, asymétrie, taille) telle que vue par l'algorithme. \n")

                if diam < 80 and comp > 0.70:
                    niveau = "faible"
                    st.write("ℹ️ La lésion est de petite taille et présente une forme globalement régulière.")
                    st.write("   Le score de complexité doit être interprété avec prudence.")
                else:
                    if suspicion_score >= 70:
                        niveau = "élevé"
                    elif suspicion_score >= 40:
                        niveau = "modéré"
                    else:
                        niveau = "faible"

                if niveau == "élevé":
                    st.write("🔎 **Niveau de complexité morphologique : ÉLEVÉ**")
                    st.write("   La lésion présente plusieurs caractéristiques géométriques atypiques")
                    st.write("   (asymétrie, bords irréguliers et/ou grande taille).")
                elif niveau == "modéré":
                    st.write("🔎 **Niveau de complexité morphologique : MODÉRÉ**")
                    st.write("   Certaines caractéristiques géométriques sont un peu atypiques, ")
                    st.write("   mais cela ne signifie PAS qu'il s'agit d'une lésion dangereuse.")
                else:
                    st.write("🔎 **Niveau de complexité morphologique : FAIBLE**")
                    st.write("   La lésion apparaît géométriquement simple et plutôt régulière.")

                st.write("\n❗ **Important :**")
                st.write("   - Ce score n'est PAS un diagnostic de bénignité ou de malignité.")
                st.write("   - Il ne remplace en aucun cas l'avis d'un dermatologue.")
                st.write("   - Un nævus bénin peut avoir un score élevé, et inversement.")

        plt.close('all')
        st.caption("🧪 Prototype MLOps. Aucun diagnostic médical.")