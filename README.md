# CEBT: Counterfactual Event Bottleneck Transformer

CEBT is a research codebase for a new architecture for event-driven financial modeling. The central claim is not that another SEC benchmark is needed. The claim is architectural:

> Financial event models should separate ordinary market dynamics from event-induced causal residuals. CEBT learns a compact stochastic bottleneck for the incremental effect of a disclosure event, trained with matched no-event controls, counterfactual consistency losses, and leakage-safe market outcomes.

The first experimental target is SEC disclosure events, especially 8-K filings, because they have sharp public timestamps and observable post-event market reactions.

## Why This Is Novel

CEBT is positioned against event-driven forecasting and temporal leakage work:

- CAMEF uses causal-augmented multi-modal financial forecasting, but CEBT makes the event-induced residual an explicit stochastic bottleneck.
- Causal Transformer and G-Transformer estimate counterfactual outcomes over time, but CEBT targets disclosure events with no-event dynamics plus event residual decomposition.
- Chronos and other time-series foundation models model sequences, but do not isolate event-induced deltas.
- Profit Mirage studies leakage in financial agents; CEBT builds leakage resistance into feature construction and training.
- Subliminal-learning work motivates auditing whether generated summaries or embeddings encode hidden label signal.

## Repository Map

```text
configs/              Experiment and model configs
docs/                 Research notes, literature map, and paper outline
src/cebt/data/        SEC events, public prices, trading calendar
src/cebt/features/    Leakage-safe windows, labels, controls, embeddings
src/cebt/models/      NoEventDynamics, EventBottleneck, residual heads, baselines
src/cebt/training/    Dataset loading, losses, training loop
src/cebt/evaluation/  Metrics, leakage validation, diagnostics
src/cebt/analysis/    Table generation helpers
scripts/              Config-driven pipeline entrypoints
tests/                Unit and smoke tests
paper/                Paper draft assets
data/                 Local data cache; generated artifacts are gitignored
```

## Setup

Use Python 3.11 through `uv`:

```bash
uv sync --extra dev
cp .env.example .env
```

Set `SEC_USER_AGENT` in `.env` or your shell before downloading SEC data. The project uses only public/free data sources and open-source tooling.

## Reproduction Path

Start with the code smoke test:

```bash
uv run pytest
uv run python scripts/00_smoke.py --config configs/pilot.yaml --output-dir data/runs/smoke
```

Then run a small real-data pilot:

```bash
uv run python scripts/10_build_events.py --config configs/pilot.yaml --output-dir data/processed/pilot
uv run python scripts/20_download_prices.py --config configs/pilot.yaml --output-dir data/processed/pilot
uv run python scripts/30_build_features.py --config configs/pilot.yaml --output-dir data/processed/pilot
uv run python scripts/40_train.py --config configs/pilot.yaml --output-dir data/runs/pilot
uv run python scripts/50_eval.py --config configs/pilot.yaml --output-dir data/runs/pilot
uv run python scripts/60_ablate.py --config configs/pilot.yaml --output-dir data/runs/pilot
uv run python scripts/70_make_tables.py --config configs/pilot.yaml --output-dir data/runs/pilot
```

For the paper-scale v1 run used by the current draft:

```bash
SEC_USER_AGENT="CEBT academic research <name> <email>" uv run python scripts/10_build_events.py --config configs/paper.yaml --output-dir data/processed/paper_v1
uv run python scripts/20_download_prices.py --config configs/paper.yaml --output-dir data/processed/paper_v1
uv run python scripts/30_build_features.py --config configs/paper.yaml --output-dir data/processed/paper_v1
uv run python scripts/60_ablate.py --config configs/paper.yaml --output-dir data/runs/paper_v1
uv run python scripts/70_make_tables.py --config configs/paper.yaml --output-dir data/runs/paper_v1
```

The current draft lives at `paper/main.tex`, with figures in `paper/figures/` and table exports in `paper/tables/`.

## Data Policy

CEBT uses public SEC filings and public market prices. Price downloads try Stooq first and fall back to Yahoo's public chart endpoint when Stooq requires an API key. It does not use paid data, closed-source model APIs, fake research labels, or generated results as evidence. Unit tests use minimal synthetic tensors only to verify code behavior; experiments and reported results must use real downloaded data.

This project is for research only and is not investment advice.
