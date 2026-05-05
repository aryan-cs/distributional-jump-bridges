# Distributional Jump Bridges for Disclosure Response Forecasts

This repository studies event-driven financial modeling as a latent distribution-transport problem.
The current paper centers on **Distributional Jump Bridges (DJB)** and analyzes a
return-conservative variant as a targeted diagnostic:

> Public disclosures should transport response risk, liquidity, and uncertainty, but should not
> directly overwrite the no-event abnormal-return mean unless the evidence supports that path.
> The bridge improves leakage-safe response-regression error over fusion baselines, while the
> return-conservative variant isolates the effect of blocking direct return-mean transport.

The experimental target is SEC 8-K disclosures because they have public accepted timestamps and
observable post-event market responses. The codebase keeps several internal baselines for controlled
comparison, including fusion, bottleneck, operator, state-jump, and unconstrained bridge variants.

## Why This Is Novel

DJB is positioned against event-driven forecasting and temporal leakage work:

- CAMEF uses causal-augmented multi-modal financial forecasting, but DJB makes the disclosure
  effect a constrained transport from a no-event distribution to an event-response distribution.
- Causal Transformer and G-Transformer estimate counterfactual outcomes over time, while DJB targets
  observed SEC disclosure timestamps and explicitly decomposes no-event dynamics from disclosure
  response transport.
- State-space and Koopman-style models learn temporal dynamics, but typically do not test whether
  disclosure text should be blocked from directly shifting return means.
- Profit Mirage studies leakage in financial agents; this repo builds timestamp gates into feature
  construction and evaluation.

## Repository Map

```text
configs/              Experiment and model configs
docs/                 Research notes, literature map, and paper outline
src/                 Python package for data, features, models, training, evaluation, and analysis
scripts/              Config-driven pipeline entrypoints
tests/                Unit and smoke tests
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
SEC_USER_AGENT="DJB academic research <name> <email>" uv run python scripts/10_build_events.py --config configs/paper_v3.yaml --output-dir data/processed/paper_v3
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

Then build the BGE-small disclosure embeddings and train the headline DJB run:

```bash
uv run python scripts/30_build_features.py --config configs/paper_v3_bge.yaml --output-dir data/processed/paper_v3_bge
uv run python scripts/40_train.py --config configs/paper_v3_bge_djb.yaml --model-name djb --output-dir data/runs/paper_v3_bge_djb_best
uv run python scripts/50_eval.py --config configs/paper_v3_bge_djb.yaml --model-name djb --output-dir data/runs/paper_v3_bge_djb_best
uv run python scripts/50_eval.py --config configs/paper_v3_bge_djb.yaml --model-name djb --output-dir data/runs/paper_v3_bge_djb_best --intervention no_jump
uv run python scripts/50_eval.py --config configs/paper_v3_bge_djb.yaml --model-name djb --output-dir data/runs/paper_v3_bge_djb_best --intervention zero_event
uv run python scripts/50_eval.py --config configs/paper_v3_bge_djb.yaml --model-name djb --output-dir data/runs/paper_v3_bge_djb_best --intervention shuffle_event
```

Train the return-conservative diagnostic variant with:

```bash
uv run python scripts/40_train.py --config configs/paper_v3_bge_rc_djb.yaml --model-name rc_djb --output-dir data/runs/paper_v3_bge_rc_djb_best
uv run python scripts/50_eval.py --config configs/paper_v3_bge_rc_djb.yaml --model-name rc_djb --output-dir data/runs/paper_v3_bge_rc_djb_best
uv run python scripts/50_eval.py --config configs/paper_v3_bge_rc_djb.yaml --model-name rc_djb --output-dir data/runs/paper_v3_bge_rc_djb_best --intervention no_jump
uv run python scripts/50_eval.py --config configs/paper_v3_bge_rc_djb.yaml --model-name rc_djb --output-dir data/runs/paper_v3_bge_rc_djb_best --intervention zero_event
uv run python scripts/50_eval.py --config configs/paper_v3_bge_rc_djb.yaml --model-name rc_djb --output-dir data/runs/paper_v3_bge_rc_djb_best --intervention shuffle_event
```

For the BGE state-jump comparison run:

```bash
uv run python scripts/40_train.py --config configs/paper_v3_bge_ejssm.yaml --model-name ejssm --output-dir data/runs/paper_v3_bge_ejssm_balanced
uv run python scripts/50_eval.py --config configs/paper_v3_bge_ejssm.yaml --model-name ejssm --output-dir data/runs/paper_v3_bge_ejssm_balanced
uv run python scripts/50_eval.py --config configs/paper_v3_bge_ejssm.yaml --model-name ejssm --output-dir data/runs/paper_v3_bge_ejssm_balanced --intervention no_jump
uv run python scripts/50_eval.py --config configs/paper_v3_bge_ejssm.yaml --model-name ejssm --output-dir data/runs/paper_v3_bge_ejssm_balanced --intervention zero_event
uv run python scripts/50_eval.py --config configs/paper_v3_bge_ejssm.yaml --model-name ejssm --output-dir data/runs/paper_v3_bge_ejssm_balanced --intervention shuffle_event
```

The current paper-scale run uses 7,236 real SEC 8-K events, 7,236 matched no-event controls, and
2,463 held-out rows. The draft reports that DJB improves event-response MSE over concat and
operator-fusion baselines, and uses the return-conservative variant to audit how blocking direct
return-mean transport changes the learned response distribution. It does not claim to be a trading
system or to dominate every MSE-only ablation.

The current draft lives at `docs/main.tex`, the compiled PDF is `docs/main.pdf`, figures are in
`docs/figures/`, and table exports are in `docs/tables/`.

The GitHub repository contains source code, configs, tests, table/figure scripts, and paper source.
Generated artifacts such as processed metadata, downloaded prices, checkpoints, and cached
predictions are intentionally excluded from Git and should be regenerated with the commands above.

## Data Policy

The project uses public SEC filings and public market prices. Price downloads try Stooq first and
fall back to Yahoo's public chart endpoint when needed. It does not use paid data, closed-source
model APIs, fake research labels, or generated results as evidence. Unit tests use minimal
synthetic tensors only to verify code behavior; experiments and reported results must use real
downloaded data.

This project is for research only and is not investment advice.
