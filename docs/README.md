# Paper

This directory contains the current DJB paper draft.

- `main.tex` is the LaTeX source.
- `main.pdf` is the current compiled paper snapshot.
- `djb.pdf` is a submitter-friendly copy of the compiled paper.
- `references.bib` contains the bibliography.
- `figures/` contains generated paper figures copied from the paper-scale run.
- `tables/` contains generated CSV tables copied from the paper-scale run.

The current draft reports the `configs/paper_v3.yaml` run: 98 usable large
U.S. firms, 7,236 real SEC 8-K events, 7,236 matched controls, and 2,463
held-out rows.

Build the PDF with:

```bash
cd docs && tectonic main.tex
```

LaTeX intermediates are intentionally gitignored. The manuscript source, compiled
snapshot, tables, and figures are tracked.
