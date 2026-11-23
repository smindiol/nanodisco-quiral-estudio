#!/usr/bin/env bash
# Publica main:/ como página de lectura. Usa la autenticación existente de gh.
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
mkdir -p "$repo_dir/build"
repo_slug="$(gh repo view --json nameWithOwner --jq .nameWithOwner)"
if gh api "repos/$repo_slug/pages" > "$repo_dir/build/pages.json" 2> "$repo_dir/build/pages_error.log"; then
  gh api --method PUT "repos/$repo_slug/pages" \
    -f 'build_type=legacy' -f 'source[branch]=main' -f 'source[path]=/'
elif grep -q 'HTTP 404' "$repo_dir/build/pages_error.log"; then
  gh api --method POST "repos/$repo_slug/pages" \
    -f 'build_type=legacy' -f 'source[branch]=main' -f 'source[path]=/'
else
  cat "$repo_dir/build/pages_error.log" >&2
  exit 1
fi
gh api "repos/$repo_slug/pages" --jq .html_url
printf 'GitHub debe completar la publicación antes de que la página esté disponible.\n'
