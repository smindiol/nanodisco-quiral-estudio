#!/usr/bin/env bash
# Tectonic con recursos locales; pdfLaTeX como alternativa si el bundle no está disponible.
set -euo pipefail
source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$source_dir/.." && pwd)"
build_dir="$repo_dir/build/presentacion"
mkdir -p "$build_dir"
cd "$source_dir"
tectonic_bin="$(command -v tectonic || true)"
if [[ -z "$tectonic_bin" && -x /opt/antigravity/tectonic ]]; then
  tectonic_bin=/opt/antigravity/tectonic
fi
if [[ -n "$tectonic_bin" ]] && XDG_CACHE_HOME="$repo_dir/build/tectonic-cache" "$tectonic_bin" --only-cached --keep-logs --outdir "$build_dir" presentacion.tex > "$build_dir/tectonic.log" 2>&1; then
  printf 'Compilación con Tectonic completada.\n'
else
  printf 'Se usa pdfLaTeX; el registro de Tectonic queda en build/presentacion/.\n'
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$build_dir" presentacion.tex > "$build_dir/pdflatex_1.log" 2>&1
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$build_dir" presentacion.tex > "$build_dir/pdflatex_2.log" 2>&1
fi
cp "$build_dir/presentacion.pdf" "$source_dir/presentacion.pdf"
printf 'Presentación compilada: presentacion/presentacion.pdf\n'
