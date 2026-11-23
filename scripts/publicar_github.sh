#!/usr/bin/env bash
# Publica el repositorio preparado usando la sesión del propio usuario.
# No copia ni almacena tokens. No reemplaza un remoto existente.
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
if [[ -n "$(git status --porcelain)" ]]; then
  printf 'Hay cambios sin commit. Revise y confirme esos archivos antes de publicar.\n' >&2
  exit 1
fi
login="$(gh api user --jq .login)"
repo_name="nanodisco-quiral-estudio"
if git remote get-url origin >/dev/null 2>&1; then
  printf 'El remoto origin ya existe; no se modifica ni se publica automáticamente.\n' >&2
  exit 1
fi
gh repo create "$login/$repo_name" --public --source=. --remote=origin --push \
  --description 'Nanodisco quiral: simulación, pruebas estadísticas y explicación matemática de ocho estados'
gh repo view "$login/$repo_name" --json url --jq .url
