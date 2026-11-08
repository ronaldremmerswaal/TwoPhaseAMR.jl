# Requires latexmk, pdflatex, BibTeX and the packages imported by techref.tex.
# `figures` needs Julia; the PDF build only reads the committed figure data.
.PHONY: all clean figures check
all:
	latexmk -pdf -interaction=nonstopmode -halt-on-error techref.tex

figures:
	OPENBLAS_NUM_THREADS=1 julia --project=figures -e 'using Pkg; Pkg.instantiate()'
	OPENBLAS_NUM_THREADS=1 julia --project=figures figures/generate.jl

check:
	python3 check_links.py

clean:
	latexmk -C techref.tex
	rm -f techref.lod
