# EJSSM: Event-Jump State Space Models for Financial Disclosures

This repository studies event-driven financial modeling as a latent dynamics problem. The current
paper centers on the **Event-Jump State Space Model (EJSSM)**:

> Public disclosures can be modeled as bounded jumps in a latent market state. Separating ordinary
> no-event evolution from disclosure-induced jumps improves leakage-safe event-response modeling
> and produces inspectable jump diagnostics.

The experimental target is SEC 8-K disclosures because they have public accepted timestamps and
observable post-event market responses. Earlier CEBT and DOT architectures remain in the codebase
as baselines.

## Why This Is Novel

EJSSM is positioned against event-driven forecasting and temporal leakage work:

- CAMEF uses causal-augmented multi-modal financial forecasting, but EJSSM makes the disclosure
  effect an explicit jump in latent state dynamics.
- Causal Transformer and G-Transformer estimate counterfactual outcomes over time, while EJSSM
  targets observed SEC disclosure timestamps and event-response distributions.
- State-space and Koopman-style models learn temporal dynamics, but typically do not condition
  latent jumps on disclosure text.
- Profit Mirage studies leakage in financial agents; this repo builds timestamp gates into feature
  construction and evaluation.

## Repository Map

```text
configs/              Experiment and model configs
docs/                 Research notes, literature map, and paper outline
src/cebt/data/        SEC events, public prices, trading calendar
src/cebt/features/    Leakage-safe windows, labels, controls, embeddings
src/cebt/models/      EJSSM, CEBT, DOT, fusion, text-only, no-event baselines
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

For the paper-scale run used by the current draft:

```bash
SEC_USER_AGENT="CEBT academic research <name> <email>" uv run python scripts/10_build_events.py --config configs/paper_v3.yaml --output-dir data/processed/paper_v3
uv run python scripts/20_download_prices.py --config configs/paper_v3.yaml --output-dir data/processed/paper_v3
uv run python scripts/30_build_features.py --config configs/paper_v3.yaml --output-dir data/processed/paper_v3
uv run python scripts/60_ablate.py --config configs/paper_v3.yaml --output-dir data/runs/paper_v3
uv run python scripts/40_train.py --config configs/paper_v3_ejssm.yaml --model-name ejssm --output-dir data/runs/paper_v3_ejssm_textjump
uv run python scripts/50_eval.py --config configs/paper_v3_ejssm.yaml --model-name ejssm --output-dir data/runs/paper_v3_ejssm_textjump
uv run python scripts/50_eval.py --config configs/paper_v3_ejssm.yaml --model-name ejssm --output-dir data/runs/paper_v3_ejssm_textjump --intervention no_jump
uv run python scripts/50_eval.py --config configs/paper_v3_ejssm.yaml --model-name ejssm --output-dir data/runs/paper_v3_ejssm_textjump --intervention zero_event
uv run python scripts/50_eval.py --config configs/paper_v3_ejssm.yaml --model-name ejssm --output-dir data/runs/paper_v3_ejssm_textjump --intervention shuffle_event
uv run python scripts/70_make_tables.py --config configs/paper_v3.yaml --output-dir data/runs/paper_v3
```

The current paper-scale run uses 7,236 real SEC 8-K events, 7,236 matched no-event controls, and
2,463 held-out rows. The draft reports that EJSSM improves event-response MSE and passes
no-jump, zero-disclosure, and shuffled-disclosure MSE stress tests; it does not claim to solve
abnormal-return ranking or to dominate every probabilistic-likelihood ablation.

The current draft lives at `paper/main.tex`, the compiled PDF is `paper/main.pdf`, figures are in
`paper/figures/`, and table exports are in `paper/tables/`.

## Data Policy

The project uses public SEC filings and public market prices. Price downloads try Stooq first and
fall back to Yahoo's public chart endpoint when needed. It does not use paid data, closed-source
model APIs, fake research labels, or generated results as evidence. Unit tests use minimal
synthetic tensors only to verify code behavior; experiments and reported results must use real
downloaded data.

This project is for research only and is not investment advice.
