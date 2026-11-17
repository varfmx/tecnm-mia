#!/usr/bin/env bash
# Compila una actividad y deja solo el PDF.
# Uso: ./compilar.sh act02   (compila act02/act02.tex)
# tecnm.sty y referencias.bib están en este directorio (actividades/).
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Uso: $0 <carpeta>   (ej. $0 act02)" >&2
  exit 1
fi

RAIZ="$(cd "$(dirname "$0")" && pwd)"
DIR="${1%/}"
DOC="$(basename "$DIR")"

cd "$RAIZ/$DIR"
if [ ! -f "$DOC.tex" ]; then
  echo "No existe $DIR/$DOC.tex" >&2
  exit 1
fi

export TEXINPUTS="$RAIZ:"
export BIBINPUTS="$RAIZ:"

# -bibtex: fuerza biber aunque latexmk no encuentre el .bib en el directorio actual
latexmk -g -bibtex -pdf -interaction=nonstopmode -halt-on-error "$DOC.tex"

# Borra auxiliares (incluye .bbl), conserva el PDF
latexmk -c -bibtex "$DOC.tex"
rm -f "$DOC.run.xml" "$DOC.bcf" "$DOC.synctex.gz" "$DOC.listing"

echo "Listo: $DIR/$DOC.pdf"
