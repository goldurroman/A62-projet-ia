
# 📝 Guide de Synchronisation des Données (DVC + Kubernetes)

Ce guide résume la procédure opérationnelle pour mettre à jour le jeu de données (dataset), reconstruire l'image Docker de manière optimisée et piloter le Job de synchronisation sur le cluster Kubernetes.

---

## 🛠️ Architecture du Pipeline

Le script `sync.py` gère de manière autonome l'alignement entre trois entités :
1. **L'image Docker** (qui embarque la version du dataset au moment du build).
2. **Le Volume Persistant (PVC)** dans Kubernetes (où le modèle va lire les images).
3. **Le Stockage Distant MinIO** (qui sert de serveur de stockage centralisé pour DVC).

Le script est **100 % idempotent** : il détecte automatiquement si le stockage local est corrompu ou obsolète, nettoie l'environnement si nécessaire, et réaligne les données sans intervention humaine.

---

## Procédure Standard : Ajout de Nouvelles Images

Suis rigoureusement ces 4 étapes lorsque tu ajoutes de nouvelles images ou modifies des étiquettes (labels) dans ton dossier local `data/`.

### Étape 1 : Ajout des images et Indexation locale avec DVC
Ajoute les nouvelles images et labels dans le dossier data/train et/ou data/val.

Ajoute les nouveaux fichiers au suivi de DVC pour mettre à jour le fichier d'empreinte (`data.dvc`) :

## powershell
### dvc add data
(Note : Si l'environnement virtuel menv-dvc n'est pas activé ou si le plugin S3 rencontre une erreur en local, tu peux ignorer l'étape du dvc push local. Le Job Kubernetes s'en chargera automatiquement grâce au mécanisme de secours).

### Étape 2 : Build de l'image Docker (Version Optimisée)
Compile la nouvelle image en incrémentant le tag de version (ex: v13).

⚠️ Règle d'or : N'utilise plus le flag --no-cache. En l'enlevant, Docker réutilise les couches existantes (comme l'installation lourde de PyTorch/Ultralytics) et ton build passera de 10 minutes à moins de 5 secondes.

## PowerShell
### docker build -t romangoldur/a62-sync:v13 -f docker/sync.Dockerfile .

### Étape 3 : Mise à jour du Manifeste Kubernetes
Ouvre ton fichier **k8s/dataset-sync-job.yaml** et modifiez le tag de l'image pour cibler la nouvelle version :

```YAML
spec:
  template:
    spec:
      containers:
      - name: sync-container
        image: romangoldur/a62-sync:v13  # <--- Mettre à jour le tag ici
```
### Étape 4 : Déploiement et Suivi sur le Cluster
Exécute cette séquence dans ton terminal pour appliquer les changements :



PowerShell
## 1. Supprimer l'ancien Job pour libérer le nom dans le cluster
### kubectl delete job dataset-sync --ignore-not-found=true

## 2. Créer le nouveau Job de synchronisation
### kubectl apply -f k8s/dataset-sync-job.yaml

## 3. Suivre les logs en temps réel pour valider le comportement
### kubectl logs -f job/dataset-sync

# Comment Interpréter les Logs de Validation ?

## Cas A : Le Dataset a changé (Nouvelles images détectées)
Le comportement attendu doit être le suivant :

Le dvc pull initial échoue avec une alerte indiquant que le nouveau hash .dir est introuvable sur le cloud.

Le script capture l'erreur et affiche : [WARNING] Cache distant manquant ou incomplet. Réalignement complet du workspace....

La commande rm -rf nettoie proprement le volume.

Le dvc push final téléverse uniquement les nouveautés et affiche le nombre de fichiers modifiés (ex: 3 files pushed).

Le job se termine par : FIN DU JOB : EXÉCUTION TERMINÉE AVEC SUCCÈS.

## Cas B : Le Dataset n'a pas bougé (Relance ou sécurité)
Si le dataset est déjà à jour sur le volume et sur MinIO :

Le dvc pull réussit immédiatement.

Le log affiche : [STDOUT] Everything is up to date.

Le bloc de nettoyage et de transfert local est ignoré, garantissant une exécution instantanée.

# Nettoyage Post-Synchronisation
Une fois que les logs confirment le statut SUCCÈS, le volume persistant (PVC) est officiellement prêt pour l'entraînement. Les pods terminés consommant des ressources d'historique dans le cluster, supprime le job de synchronisation :


### kubectl delete job dataset-sync