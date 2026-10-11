#!/bin/bash
# Build the manuscript PDF from main.tex.
# Regenerates a note-free copy of the shared references.bib, then runs the
# standard latex -> bibtex -> latex -> latex sequence. Requires pdflatex + bibtex.
set -e
cd "$(dirname "$0")"

true  # bibliography ships note-free as references.bib

# Convert SVG figure sources to PDF (house figure standard; docs/figure-standards.md).
for svg in fig-*.svg; do
  [ -e "$svg" ] || continue
  rsvg-convert -f pdf -o "${svg%.svg}.pdf" "$svg"
done

pdflatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null
bibtex main > /dev/null || true
pdflatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null
pdflatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null

echo "Built: $(pwd)/main.pdf"
