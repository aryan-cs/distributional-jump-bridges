# Paper

This directory contains the current CEBT paper draft.

- `main.tex` is the LaTeX source.
- `main.pdf` is the current compiled paper snapshot.
- `references.bib` contains the bibliography.
- `figures/` contains generated paper figures copied from the paper-scale run.
- `tables/` contains generated CSV tables copied from the paper-scale run.

The current draft reports the `configs/paper_v3.yaml` run: 98 usable large
U.S. firms, 7,236 real SEC 8-K events, 7,236 matched controls, and 2,463
held-out rows.

Build the PDF with:

```bash
tectonic paper/main.tex --outdir paper
```

LaTeX intermediates are intentionally gitignored. The paper source, compiled
snapshot, tables, and figures are tracked.
