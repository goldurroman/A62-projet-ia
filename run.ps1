Write-Host "=== [MLOps Pipeline] Matrice d'entrainement Serialisee ===" -ForegroundColor Cyan

# 1. Comptage du dataset
$TRAIN_COUNT = (Get-ChildItem -Path "C:\temp\A62-projet-ia\data\train\images" -Filter *.jpg).Count
$VAL_COUNT   = (Get-ChildItem -Path "C:\temp\A62-projet-ia\data\val\images" -Filter *.jpg).Count

Write-Host "[DATASET] Train=$TRAIN_COUNT | Val=$VAL_COUNT" -ForegroundColor Green

# 2. Matrice
$MODELS       = @("yolov8n-seg.pt") # yolov8s-seg.pt, yolov8m-seg.pt
$IMG_SIZES    = @("384") #  128 256 384 512 640 768
$COLOR_MODES  = @("gray", "rgb") # gray, rgb
$EPOCHS_LIST  = @("2") # 15 20 50 100 150 200

if (!(Test-Path "k8s/generated")) { New-Item -ItemType Directory -Path "k8s/generated" | Out-Null }

foreach ($model in $MODELS) {
    foreach ($size in $IMG_SIZES) {
        foreach ($color in $COLOR_MODES) {
            foreach ($epoch in $EPOCHS_LIST) {

                $model_short = $model.Replace("-seg.pt", "").Replace(".pt", "")
                $SEMANTIC_IDENTIFIER = "${model_short}-${TRAIN_COUNT}-${VAL_COUNT}-${size}-${color}-${epoch}"

                Write-Host "`n--------------------------------------------------------" -ForegroundColor Gray
                Write-Host "[NEXT] $SEMANTIC_IDENTIFIER" -ForegroundColor Yellow

                # A. Generation du YAML
                $template = Get-Content "k8s/train-job.yaml" -Raw
                $template = $template -replace '\${JOB_NAME}', $SEMANTIC_IDENTIFIER
                $template = $template -replace '\${CONTAINER_NAME}', $SEMANTIC_IDENTIFIER
                $template = $template -replace '\${MODEL_NAME}', $model
                $template = $template -replace '\${IMG_SIZE}', $size
                $template = $template -replace '\${COLOR_MODE}', $color
                $template = $template -replace '\${EPOCHS}', $epoch

                $target_file = "k8s/generated/job-${SEMANTIC_IDENTIFIER}.yaml"
                Set-Content -Path $target_file -Value $template

                # B. Deploiement
                Write-Host "[K8S] Lancement du Job..." -ForegroundColor Blue
                kubectl apply -f $target_file | Out-Null
                Write-Host "[DEBUG] Image utilisee par Kubernetes :" -ForegroundColor Yellow
                kubectl get pods -l job-name=job-${SEMANTIC_IDENTIFIER} -o jsonpath='{.items[0].spec.containers[*].image}'

                # Attente du Pod Ready
                kubectl wait --for=condition=Ready pod -l job-name=job-${SEMANTIC_IDENTIFIER} --timeout=120s
                Write-Host "[DEBUG] Etat du pod :" -ForegroundColor Yellow
                kubectl get pod -l job-name=job-${SEMANTIC_IDENTIFIER} -o wide

                # Recuperation du Pod
                $POD_NAME = (kubectl get pods -l job-name=job-${SEMANTIC_IDENTIFIER} -o jsonpath='{.items[0].metadata.name}')

                Write-Host "[DEBUG] Logs du pod (debut) :" -ForegroundColor Yellow

                Write-Host "[LOGS] Pod=$POD_NAME" -ForegroundColor Gray
                kubectl logs $POD_NAME -c $SEMANTIC_IDENTIFIER -f

                # C. Attente de fin du Job
                Write-Host "[K8S] Attente de completion..." -ForegroundColor Gray
                $job_status = kubectl wait --for=condition=complete job/job-${SEMANTIC_IDENTIFIER} --timeout=3600s 2>&1

                # D. Export logs si echec
                if ($job_status -like "*error*") {
                    Write-Host "[ERROR] Job echoue : $SEMANTIC_IDENTIFIER" -ForegroundColor Red
                    $LOG_FILE = "k8s/generated/logs-${SEMANTIC_IDENTIFIER}.txt"
                    kubectl logs $POD_NAME -c $SEMANTIC_IDENTIFIER | Out-File -FilePath $LOG_FILE -Encoding utf8
                    Write-Host "[DEBUG] Logs exportes proprement (UTF-8) : $LOG_FILE" -ForegroundColor Yellow
                }
                Write-Host "[DEBUG] Job termine. Verification MLflow..." -ForegroundColor Yellow

                # E. Nettoyage
                Write-Host "[CLEANUP] Suppression du Job..." -ForegroundColor Green
                kubectl delete job job-${SEMANTIC_IDENTIFIER} --cascade=foreground | Out-Null

                Start-Sleep -Seconds 5
            }
        }
    }
}

Write-Host "`n=== [MLOps Pipeline] Matrice terminee ===" -ForegroundColor Cyan
