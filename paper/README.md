# Paper

This directory contains the current CEBT paper draft.

- `main.tex` is the LaTeX source.
- `main.pdf` is the current compiled paper snapshot.
- `references.bib` contains the bibliography.
- `figures/` contains generated paper figures copied from the paper-scale run.
- `tables/` contains generated CSV tables copied from the paper-scale run.

The current draft reports the `configs/paper_v2.yaml` run: 50 large U.S. firms,
2,971 real SEC 8-K events, 2,971 matched controls, and 1,220 held-out rows.

Build the PDF with:

```bash
tectonic paper/main.tex --outdir paper
```

LaTeX intermediates are intentionally gitignored. The paper source, compiled
snapshot, tables, and figures are tracked.
