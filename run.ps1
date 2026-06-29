Write-Host "[INFO - run.ps1] - build de l'image"
docker build -t projet-synthese .

Write-Host "[INFO - run.ps1] Generation du nom du conteneur..."
$CONTAINER_NAME = python gen_container_name.py

Write-Host "[INFO - run.ps1] Nom obtenu : $CONTAINER_NAME"
Write-Host "[INFO - run.ps1] Lancement du conteneur Docker avec ce nom..."

docker run --name $CONTAINER_NAME `
    -v C:\temp\A62-projet-ia\data:/app/data `
    -v C:\temp\A62-projet-ia\mlruns:/app/mlruns `
    -v C:\temp\A62-projet-ia\mlflow.db:/app/mlflow.db `
    projet-synthese