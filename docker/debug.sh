#!/bin/sh
echo "=== DEBUG: Répertoire de travail ==="
pwd

echo "=== DEBUG: Contenu de /app ==="
ls -R /app

echo "=== DEBUG: Contenu de /workspace ==="
ls -R /workspace 2>/dev/null || echo "/workspace introuvable"

echo "=== DEBUG: Variables d'environnement ==="
env

echo "=== DEBUG: Fin du debug ==="