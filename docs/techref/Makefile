# Requires latexmk, pdflatex, BibTeX and the packages imported by techref.tex.
.PHONY: all clean
all:
	latexmk -pdf -interaction=nonstopmode -halt-on-error techref.tex

clean:
	latexmk -C techref.tex
	rm -f techref.lod
