#!/usr/bin/env python3
import os
import sys
import logging
import subprocess
import urllib.request
import urllib.error

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

BUCKET_NAME = "dvc-remote"
ENDPOINT_URL = "http://minio-service:9000"
WORKSPACE_DIR = "/workspace"


def run(cmd, cwd=None):
    """Exécute une commande système de manière sécurisée."""
    logging.info(f"[EXEC] {cmd} (dans {cwd or '.'})")
    try:
        result = subprocess.run(
            cmd, shell=True, check=True, cwd=cwd,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if result.stdout.strip():
            for line in result.stdout.splitlines():
                logging.info(f"[STDOUT] {line}")
        return True
    except subprocess.CalledProcessError as e:
        if e.stderr.strip():
            for line in e.stderr.splitlines():
                logging.error(f"[STDERR] {line}")
        logging.error(f"[ÉCHEC] Code de retour : {e.returncode} pour la commande : {cmd}")
        raise RuntimeError(f"Command failed: {cmd}")


def verifier_reseau():
    """Valide l'accessibilité de l'API MinIO."""
    logging.info(f"[RÉSEAU] Test de connexion vers {ENDPOINT_URL}...")
    try:
        with urllib.request.urlopen(ENDPOINT_URL, timeout=5) as response:
            pass
    except urllib.error.HTTPError as e:
        # MinIO renvoie souvent 400 ou 403 sur la racine sans auth, ce qui valide sa présence
        pass
    except Exception as e:
        logging.error(f"[RÉSEAU] Impossible de joindre MinIO : {e}")
        sys.exit(1)
    logging.info("[RÉSEAU] Connexion réseau validée.")


def garantir_bucket_minio():
    """Assure l'existence du bucket dvc-remote via s3fs (garanti présent avec dvc-s3)."""
    logging.info(f"[MINIO] Validation de l'existence du bucket '{BUCKET_NAME}'...")
    try:
        import s3fs
    except ImportError:
        logging.error("[ERREUR] La dépendance 's3fs' (requise par dvc-s3) est introuvable.")
        sys.exit(1)

    try:
        fs = s3fs.S3FileSystem(
            key=os.getenv("AWS_ACCESS_KEY_ID", "minioadmin"),
            secret=os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin"),
            client_kwargs={'endpoint_url': ENDPOINT_URL}
        )

        if not fs.exists(BUCKET_NAME):
            logging.warning(f"[MINIO] Le bucket '{BUCKET_NAME}' n'existe pas. Création en cours...")
            fs.mkdir(BUCKET_NAME)
            logging.info(f"[MINIO] Bucket '{BUCKET_NAME}' créé avec succès.")
        else:
            logging.info(f"[MINIO] Le bucket '{BUCKET_NAME}' existe déjà.")

    except Exception as e:
        logging.error(f"[MINIO] Erreur lors de la vérification/création du bucket : {e}")
        sys.exit(1)

def initialiser_dvc():
    """Configure le dépôt DVC sur le volume persistant."""
    if not os.path.exists(os.path.join(WORKSPACE_DIR, ".dvc")):
        logging.info("[DVC] Initialisation d'un nouveau dépôt DVC...")
        run("dvc init --no-scm", cwd=WORKSPACE_DIR)
        run("dvc config cache.type hardlink,symlink", cwd=WORKSPACE_DIR)
    else:
        logging.info("[DVC] Un dépôt DVC existant a été détecté.")
        # Sécurité : On s'assure que la configuration des liens est bien active
        run("dvc config cache.type hardlink,symlink", cwd=WORKSPACE_DIR)

    # Configuration stricte du remote (Pas de cp de data.dvc ici)
    run(f"dvc remote add -d -f minio-storage s3://{BUCKET_NAME}", cwd=WORKSPACE_DIR)
    run(f"dvc remote modify minio-storage endpointurl {ENDPOINT_URL}", cwd=WORKSPACE_DIR)


def synchroniser():
    """Gère la récupération ou la publication des données."""
    # Unique endroit où l'on synchronise le fichier de suivi
    if os.path.exists("/app/data.dvc"):
        run("cp -f /app/data.dvc /workspace/", cwd=WORKSPACE_DIR)

    try:
        logging.info("[DVC] Tentative de récupération des données existantes (dvc pull)...")
        run("dvc pull -r minio-storage", cwd=WORKSPACE_DIR)
        logging.info("[DVC] Synchronisation descendante réussie (Données à jour).")
    except RuntimeError:
        logging.warning("[DVC] Cache distant manquant ou incomplet. Réalignement complet du workspace...")

        # SÉCURITÉ K8sOps : Force brute de nettoyage des structures instables
        # On supprime le dossier data, le cache ET l'index de DVC pour repartir de zéro
        run("rm -rf /workspace/data /workspace/.dvc/cache /workspace/.dvc/index", cwd=WORKSPACE_DIR)

        logging.info("[TRANSFERT] Copie des images locales de l'image Docker vers le volume persistant...")
        run("cp -rf /app/data /workspace/", cwd=WORKSPACE_DIR)

        logging.info("[DVC] Indexation locale des données transférées...")
        run("dvc add data", cwd=WORKSPACE_DIR)

        logging.info("[DVC] Téléversement des données vers le stockage MinIO...")
        run("dvc push -r minio-storage", cwd=WORKSPACE_DIR)
        logging.info("[SUCCÈS] Initialisation et synchronisation ascendante complétées.")


def main():
    logging.info("==========================================================")
    logging.info("DÉMARRAGE DU JOB DE SYNCHRONISATION AUTONOME")
    logging.info("==========================================================")
    verifier_reseau()
    garantir_bucket_minio()
    initialiser_dvc()
    synchroniser()
    logging.info("==========================================================")
    logging.info("FIN DU JOB : EXÉCUTION TERMINÉE AVEC SUCCÈS")
    logging.info("==========================================================")


if __name__ == "__main__":
    main()