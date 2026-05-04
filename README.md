# RC-DJB: Return-Conservative Disclosure Jump Bridges

This repository studies event-driven financial modeling as a latent distribution-transport problem.
The current paper centers on the **Return-Conservative Distributional Jump Bridge (RC-DJB)**:

> Public disclosures should transport response risk, liquidity, and uncertainty, but should not
> directly overwrite the no-event abnormal-return mean unless the evidence supports that path.
> A return-mean firewall recovers statistically positive held-out rank IC while preserving
> leakage-safe response-regression gains over fusion baselines.

The experimental target is SEC 8-K disclosures because they have public accepted timestamps and
observable post-event market responses. Earlier CEBT and DOT architectures remain in the codebase
as baselines; EJSSM and unconstrained DJB are stronger state/bridge baselines.

## Why This Is Novel

RC-DJB is positioned against event-driven forecasting and temporal leakage work:

- CAMEF uses causal-augmented multi-modal financial forecasting, but RC-DJB makes the disclosure
  effect a constrained transport from a no-event distribution to an event-response distribution.
- Causal Transformer and G-Transformer estimate counterfactual outcomes over time, while RC-DJB
  targets observed SEC disclosure timestamps with a hard return-mean conservation constraint.
- State-space and Koopman-style models learn temporal dynamics, but typically do not test whether
  disclosure text should be blocked from directly shifting return means.
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

For the paper-scale run used by the current draft, first build the public SEC/price substrate and
the deterministic baseline tables:

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

Then build the BGE-small disclosure embeddings and train the headline RC-DJB run:

```bash
uv run python scripts/30_build_features.py --config configs/paper_v3_bge.yaml --output-dir data/processed/paper_v3_bge
uv run python scripts/40_train.py --config configs/paper_v3_bge_rc_djb.yaml --model-name rc_djb --output-dir data/runs/paper_v3_bge_rc_djb_best
uv run python scripts/50_eval.py --config configs/paper_v3_bge_rc_djb.yaml --model-name rc_djb --output-dir data/runs/paper_v3_bge_rc_djb_best
uv run python scripts/50_eval.py --config configs/paper_v3_bge_rc_djb.yaml --model-name rc_djb --output-dir data/runs/paper_v3_bge_rc_djb_best --intervention no_jump
uv run python scripts/50_eval.py --config configs/paper_v3_bge_rc_djb.yaml --model-name rc_djb --output-dir data/runs/paper_v3_bge_rc_djb_best --intervention zero_event
uv run python scripts/50_eval.py --config configs/paper_v3_bge_rc_djb.yaml --model-name rc_djb --output-dir data/runs/paper_v3_bge_rc_djb_best --intervention shuffle_event
```

For the BGE-EJSSM comparison run:

```bash
uv run python scripts/40_train.py --config configs/paper_v3_bge_ejssm.yaml --model-name ejssm --output-dir data/runs/paper_v3_bge_ejssm_balanced
uv run python scripts/50_eval.py --config configs/paper_v3_bge_ejssm.yaml --model-name ejssm --output-dir data/runs/paper_v3_bge_ejssm_balanced
uv run python scripts/50_eval.py --config configs/paper_v3_bge_ejssm.yaml --model-name ejssm --output-dir data/runs/paper_v3_bge_ejssm_balanced --intervention no_jump
uv run python scripts/50_eval.py --config configs/paper_v3_bge_ejssm.yaml --model-name ejssm --output-dir data/runs/paper_v3_bge_ejssm_balanced --intervention zero_event
uv run python scripts/50_eval.py --config configs/paper_v3_bge_ejssm.yaml --model-name ejssm --output-dir data/runs/paper_v3_bge_ejssm_balanced --intervention shuffle_event
```

The current paper-scale run uses 7,236 real SEC 8-K events, 7,236 matched no-event controls, and
2,463 held-out rows. The draft reports that RC-DJB recovers statistically positive held-out
abnormal-return rank IC while improving event-response MSE over concat and DOT baselines. It does
not claim to be a trading system or to dominate every MSE-only ablation.

The current draft lives at `paper/main.tex`, the compiled PDF is `paper/main.pdf`, figures are in
`paper/figures/`, and table exports are in `paper/tables/`.

## Data Policy

The project uses public SEC filings and public market prices. Price downloads try Stooq first and
fall back to Yahoo's public chart endpoint when needed. It does not use paid data, closed-source
model APIs, fake research labels, or generated results as evidence. Unit tests use minimal
synthetic tensors only to verify code behavior; experiments and reported results must use real
downloaded data.

This project is for research only and is not investment advice.
