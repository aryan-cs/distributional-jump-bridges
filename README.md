# Distributional Jump Bridges for Disclosure Response Forecasts

**Aryan Gupta** · [aryan.cs.app@gmail.com](mailto:aryan.cs.app@gmail.com)

> **Paper**: [`docs/djb.pdf`](docs/djb.pdf) · **arXiv categories**: q-fin.ST, cs.LG

---

## Abstract

Financial disclosure models often combine event text with market history and predict every market-response target through one shared path. This paper proposes **Distributional Jump Bridges (DJB)**, a neural event architecture that first estimates a no-event response distribution from pre-disclosure market state, then uses SEC filing text to transport the event response through bounded shifts in target means and predictive log-variances. A return-conservative variant (**RC-DJB**) adds a return-mean constraint, providing a controlled test of whether disclosure text should directly move near-term abnormal-return point forecasts.

In a leakage-controlled public-data study with **7,236** SEC 8-K events, **7,236** matched controls, and **2,463** held-out rows from **98** large U.S. firms:

| Metric | DJB | Best Baseline |
|--------|-----|---------------|
| Multi-target MSE | **0.0552** [0.0396, 0.0773] | 0.0566 (Event-jump SSM) |
| Abnormal-return Rank IC | **0.0330** [-0.0042, 0.0710] | 0.0254 (Event-jump SSM) |
| Paired MSE improvement vs. concat fusion | 0.0114 [0.0092, 0.0135] | — |

The results support a response-transport modeling principle: disclosure text contributes more clearly to event risk, liquidity, and uncertainty than to cluster-significant return ranking.

---

## Architecture

DJB decomposes event-response prediction into two stages:

1. **No-event distribution** — A GRU encoder maps 40 pre-event trading days of price, volume, and market-relative features to a Gaussian predictive distribution over abnormal return, volatility jump, and volume jump.
2. **Distributional bridge** — SEC filing text (chunk-pooled BGE-small embeddings) generates bounded shifts in target means and predictive log-variances, transporting the no-event distribution to an event-response distribution.

RC-DJB enforces Δμ_return = 0, so text may transport volatility, volume, and uncertainty but cannot directly overwrite the abnormal-return point forecast.

### Model Variants

| Model | Key | Description |
|-------|-----|-------------|
| No-event only | `no_event` | Prices + metadata only; no text |
| Text-only MLP | `text_only` | Text + metadata; no price history |
| Concat fusion | `concat` | Standard feature concatenation |
| Event bottleneck | `bottleneck` | Stochastic latent bottleneck with KL penalty |
| Disclosure operator | `dot` | Low-rank latent-state operator |
| Event-jump SSM | `ejssm` | GRU with sparse latent state jump |
| **DJB** | `djb` | Distributional jump bridge (unconstrained) |
| **RC-DJB** | `rc_djb` | Return-conservative DJB |

---

## Key Results

### Held-Out Performance (95% Ticker-Clustered Bootstrap CIs)

| Model | MSE | MSE CI | Rank IC | Rank IC CI |
|-------|-----|--------|---------|------------|
| No-event | 0.0695 | [0.0517, 0.0933] | -0.0393 | [-0.0886, 0.0064] |
| Concat fusion | 0.0666 | [0.0499, 0.0886] | 0.0007 | [-0.0422, 0.0389] |
| Event bottleneck | 0.0696 | [0.0524, 0.0923] | -0.0295 | [-0.0752, 0.0160] |
| Disclosure operator | 0.0698 | [0.0526, 0.0926] | -0.0043 | [-0.0487, 0.0422] |
| Event-jump SSM | 0.0566 | [0.0406, 0.0788] | 0.0254 | [-0.0107, 0.0633] |
| **DJB** | **0.0552** | [0.0396, 0.0773] | **0.0330** | [-0.0042, 0.0710] |
| RC-DJB | 0.0562 | [0.0404, 0.0783] | 0.0154 | [-0.0288, 0.0559] |

### RC-DJB Intervention Tests

| Intervention | Event MSE Increase | 95% Paired CI |
|--------------|-------------------|---------------|
| No bridge | 0.0183 | [0.0165, 0.0203] |
| Zero text | 0.0037 | [0.0025, 0.0051] |
| Shuffled text | 0.0025 | [0.0014, 0.0036] |

---

## Installation

### Requirements

- Python ≥ 3.11, < 3.13
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Setup

```bash
# Clone the repository
git clone https://github.com/aryan-cs/distributional-jump-bridges.git
cd distributional-jump-bridges

# Create virtual environment and install dependencies with uv
uv sync

# Or with pip
python -m venv .venv
source .venv/bin/activate
pip install -e .

# For text embeddings (sentence-transformers / BGE)
uv sync --extra embeddings
# or: pip install -e ".[embeddings]"

# For development (pytest, ruff)
uv sync --extra dev
# or: pip install -e ".[dev]"
```

### Environment Variables

Copy the example environment file and set your SEC User-Agent:

```bash
cp .env.example .env
```

```env
# Required by SEC fair-access policy. Use your real contact information.
SEC_USER_AGENT="DJB academic research your-name your-email@example.com"

# Optional
DJB_DATA_DIR="data"
DJB_DEVICE="auto"
```

---

## Reproduction Pipeline

The full experiment is driven by numbered scripts in `scripts/`. Each script accepts `--config` (defaults to `configs/pilot.yaml`) and `--model-name` flags.

### Quick Smoke Test

```bash
python scripts/00_smoke.py --config configs/pilot.yaml
```

### Full Pipeline

```bash
# 1. Build SEC 8-K event rows from EDGAR
python scripts/10_build_events.py --config configs/paper_v3_bge.yaml

# 2. Download public daily prices (Stooq)
python scripts/20_download_prices.py --config configs/paper_v3_bge.yaml

# 3. Build leakage-safe feature tensors with BGE embeddings
python scripts/30_build_features.py --config configs/paper_v3_bge.yaml

# 4. Train a model
python scripts/40_train.py --config configs/paper_v3_bge_djb.yaml --model-name djb

# 5. Evaluate on held-out split
python scripts/50_eval.py --config configs/paper_v3_bge_djb.yaml --model-name djb

# 6. Run baseline ablations
python scripts/60_ablate.py --config configs/paper_v3_bge.yaml

# 7. Generate result tables
python scripts/70_make_tables.py --config configs/paper_v3_bge.yaml

# 8. Generate paper figures
python scripts/80_make_rc_djb_figures.py --config configs/paper_v3_bge.yaml

# 9. Generate paper tables (LaTeX-ready)
python scripts/85_make_paper_tables.py --config configs/paper_v3_bge.yaml

# 10. Replication sweeps
python scripts/90_djb_replication_sweeps.py --config configs/paper_v3_bge.yaml
```

### Configurations

| Config | Description |
|--------|-------------|
| `configs/pilot.yaml` | Small-scale test run (hashing embeddings, multi-form) |
| `configs/paper_v3_bge.yaml` | Paper-scale run (98 firms, BGE-small, 8-K only) |
| `configs/paper_v3_bge_djb.yaml` | DJB-specific overrides for paper run |
| `configs/paper_v3_bge_rc_djb.yaml` | RC-DJB-specific overrides |
| `configs/paper_v3_bge_ejssm.yaml` | Event-jump SSM baseline |
| `configs/paper_v3_bge_ff4.yaml` | Fama-French 4-factor label variant |

---

## Project Structure

```
distributional-jump-bridges/
├── configs/                    # YAML experiment configurations
│   ├── model/                  #   Model architecture configs
│   └── experiments/            #   Replication sweep configs
├── data/                       # Generated data (gitignored)
│   ├── raw/                    #   Downloaded SEC filings and price CSVs
│   ├── processed/              #   Feature bundles (.npz)
│   └── runs/                   #   Checkpoints, metrics, tables, figures
├── docs/                       # Paper manuscript and documentation
│   ├── main.tex                #   LaTeX source
│   ├── djb.pdf                 #   Compiled paper
│   ├── references.bib          #   Bibliography
│   ├── figures/                #   PDF figures used in manuscript
│   └── tables/                 #   Generated CSV tables
├── scripts/                    # Numbered pipeline scripts (00–90)
├── src/cebt/                   # Python package
│   ├── models/cebt.py          #   All model architectures (DJB, RC-DJB, baselines)
│   ├── data/                   #   SEC client, price download, Fama-French factors
│   ├── features/               #   Feature construction and text embeddings
│   ├── training/               #   Training loop, dataset, loss functions
│   ├── evaluation/             #   Evaluation metrics, bootstrap, leakage guards
│   ├── analysis/               #   Figure and table generation
│   └── utils/                  #   Config, IO, hashing, time utilities
├── tests/                      # Test suite
├── pyproject.toml              # Project metadata and dependencies
└── .env.example                # Environment variable template
```

---

## Data

All data used in this study is sourced from public APIs:

- **SEC filings**: Downloaded from [SEC EDGAR](https://www.sec.gov/edgar) using `acceptanceDateTime` timestamps and primary document URLs
- **Daily prices**: Downloaded from public price endpoints (Stooq), with SPY as the market proxy
- **Fama-French factors**: Optionally fetched for factor-model label variants

Generated data (raw filings, price CSVs, feature tensors, checkpoints, metrics) is stored locally under `data/` and is gitignored. See [`data/README.md`](data/README.md) for details.

### Leakage Safeguards

- Features use only prices strictly before the label start date
- Labels use only future returns as targets
- Matched no-event controls enforce a 10-calendar-day blackout around known events
- Chronological train/validation/held-out split (train ≤ 2023, val = 2024, test > 2024)
- Automated tests verify timestamp gating, after-hours label starts, feature leakage rejection, and evaluation leakage guards

---

## Testing

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/test_model_and_loss.py
```

The test suite covers:

- **Timestamp and leakage**: `test_time_rules.py`, `test_evaluation_leakage.py`
- **Feature construction**: `test_feature_builder.py`, `test_factors.py`
- **Model and loss**: `test_model_and_loss.py`, `test_training_smoke.py`
- **Embeddings**: `test_embeddings.py`
- **Bootstrap and tables**: `test_bootstrap_and_tables.py`
- **Configuration**: `test_config.py`

---

## Building the Paper

The LaTeX source is in `docs/`. To compile:

```bash
cd docs && tectonic main.tex
```

A pre-compiled snapshot is available at [`docs/djb.pdf`](docs/djb.pdf).

---

## Citation

```bibtex
@article{gupta2026djb,
  title   = {Distributional Jump Bridges for Disclosure Response Forecasts},
  author  = {Gupta, Aryan},
  year    = {2026},
  note    = {Preprint. Code: \url{https://github.com/aryan-cs/distributional-jump-bridges}}
}
```

---

## License

This project is released for academic research purposes. Please see the repository for licensing details.

---

## Acknowledgments

This study uses only publicly available SEC EDGAR filings and public daily price data. The results do not constitute trading recommendations. The codebase is intended to support leakage auditing, independent reproduction, and falsification of the empirical claims.
