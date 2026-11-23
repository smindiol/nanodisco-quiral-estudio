#!/usr/bin/env bash
# Solo actualiza el PDF publicado después de completar bibliografía y referencias.
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source_dir="$repo_dir/informe/latex"
build_dir="$repo_dir/build/latex"
mkdir -p "$build_dir"
cd "$source_dir"
pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$build_dir" main.tex > "$build_dir/pdflatex_1.log" 2>&1
(
  cd "$build_dir"
  BIBINPUTS="$source_dir:" bibtex main > bibtex.log 2>&1
)
pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$build_dir" main.tex > "$build_dir/pdflatex_2.log" 2>&1
pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$build_dir" main.tex > "$build_dir/pdflatex_3.log" 2>&1
cp "$build_dir/main.pdf" "$repo_dir/docs/informe_final.pdf"
printf 'Informe compilado: docs/informe_final.pdf\n'
