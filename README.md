# Distributional Jump Bridges for Disclosure Response Forecasts

> **Paper**: [`docs/djb.pdf`](docs/djb.pdf)

---

## Overview

When a public company files a document with the SEC — an 8-K, a 10-Q, etc. — the market reacts. Prices move, volume spikes, volatility shifts. Most ML models that try to predict these reactions just concatenate the filing text with recent market data and push everything through a shared prediction head. That means the same text representation is being asked to explain return ranking, risk response, and volume changes all at once.

This paper asks whether there's a better way to wire that up.

**Distributional Jump Bridges (DJB)** separates the problem into two stages:

1. **Predict what would happen with no filing** — a no-event Gaussian distribution over abnormal return, volatility jump, and volume jump, estimated from recent price and volume history alone.
2. **Use the filing text to transport that distribution** — bounded shifts to the predicted means and log-variances, so the text acts as a residual distributional operator rather than an unconstrained feature.

A return-conservative variant, **RC-DJB**, blocks the text from shifting the return mean entirely (Δμ_return = 0 by construction), while still allowing it to adjust volatility, volume, and predictive uncertainty. This makes it possible to test whether filing text is useful for predicting returns specifically, or whether its value lies in forecasting risk and liquidity response.

The experiment uses 7,236 real SEC 8-K events from 98 large U.S. firms with matched no-event controls, chronological splits, and ticker-clustered bootstrap inference. The main finding is that DJB achieves the lowest held-out multi-target MSE, and that the routing architecture — how text enters the model — matters more than whether text is included at all.

---

## Method

### Problem Setting

The event-study framework measures abnormal returns around corporate disclosures. Standard neural approaches concatenate text embeddings with market features and predict outcomes through a shared head, conflating return-signal extraction with risk/uncertainty forecasting. DJB decomposes this into a two-stage transport:

1. **No-event dynamics** *(h → μ₀, ℓ₀)*: A 2-layer GRU encodes 40 pre-event trading days of 8-dimensional features (open, high, low, close returns, volume ratio, intraday range, gap, market-relative return) plus 6-dimensional metadata (day-of-week, month-of-year, sector indicators) into a diagonal Gaussian predictive distribution.

2. **Distributional bridge** *(e → Δμ, Δℓ)*: Filing text, represented as chunk-pooled [BGE-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) embeddings (384-dim), generates:
   - A bounded latent state jump: `j = α · tanh(u(b)) ⊙ σ(v(b))` with scale α = 0.20
   - Bounded mean and log-variance deltas: `Δμ = tanh(·) · α`, `Δℓ = tanh(·) · α`
   - Final prediction: `μ₁ = μ₀ + Δμ`, `ℓ₁ = clip(ℓ₀ + Δℓ)`

### Return-Conservative Constraint (RC-DJB)

RC-DJB sets **Δμ_return ≡ 0** by construction. Filing text can still transport volatility means, volume means, and all three log-variances, but the abnormal-return point forecast stays anchored to the no-event prediction. If removing the bridge still degrades event-row MSE under this constraint, the text must be acting through risk/liquidity/uncertainty channels.

### Targets

Three targets over a 5-trading-day post-event horizon:

| Target | Definition |
|--------|------------|
| Abnormal return | SPY-adjusted cumulative return from label-start close to horizon close |
| Volatility jump | Post-event minus pre-event daily return standard deviation |
| Volume jump | Normalized post-event minus pre-event trading volume |

### Label Construction and Leakage Discipline

- **Timestamp gating**: Features use only prices with dates strictly before the label-start date. For filings accepted before 16:00 ET on a trading day, the label window starts that day; otherwise, the next trading day.
- **Matched controls**: Each real 8-K event is paired with a same-ticker no-event date, enforcing a 10-calendar-day blackout around known events.
- **Chronological split**: Train ≤ 2023-12-31, validation = 2024, held-out > 2024. No future information leaks into model selection or feature construction.
- **Return-window alignment**: Firm returns and market returns use aligned windows to avoid stale-benchmark bias.

### Training Objective

The composite loss combines seven terms:

```
L = λ_y · L_Huber  +  λ_n · L_NLL  +  λ_c · L_control  +  λ_KL · L_KL  +  λ_s · ‖z‖₁  +  λ_v · L_consistency  +  λ_r · L_rank
```

| Term | Purpose | Weight |
|------|---------|--------|
| Huber (Smooth-L1) | Target-standardized regression | w = (3.0, 1.0, 0.25) |
| Gaussian NLL | Calibrated uncertainty | 0.10 |
| Control penalty | Suppress bridge on no-event rows | 0.35 |
| KL divergence | Bottleneck regularization (bottleneck models only) | 0.0 |
| L1 sparsity | Bridge latent sparsity | 0.001 |
| Consistency | Residual variance penalty | 0.05 |
| Pairwise rank | Return-ranking objective (T=0.02) | 0.15 |

### Evaluation Protocol

- **Primary metric**: Multi-target MSE over abnormal return, volatility jump, and volume jump
- **Ranking metric**: Spearman Rank IC between predicted and realized abnormal return
- **Uncertainty**: 95% ticker-clustered bootstrap intervals (B=1000); paired comparisons use B=2000 row-paired resampling
- **Stability**: Leave-one-ticker-out (LOTO) Rank IC over all 98 held-out tickers
- **Probabilistic**: Gaussian interval coverage at 80% and 95% nominal levels

### Baselines

All models share the same data pipeline, chronological split, frozen text representation, and evaluation protocol:

| Model | Key | Routing Mechanism |
|-------|-----|-------------------|
| No-event only | `no_event` | No text; prices + metadata → prediction |
| Text-only MLP | `text_only` | No prices; text + metadata → prediction |
| Concat fusion | `concat` | Concatenate price state + text + metadata → shared head |
| Event bottleneck | `bottleneck` | Stochastic VAE-style latent with KL penalty |
| Disclosure operator | `dot` | Text → low-rank operator on latent state |
| Event-jump SSM | `ejssm` | Text → sparse latent state jump → separate outcome head |
| **DJB** | `djb` | Text → bounded distributional transport (mean + log-var) |
| **RC-DJB** | `rc_djb` | DJB with Δμ_return = 0 |

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
| **DJB** | **0.0552** | **[0.0396, 0.0773]** | **0.0330** | [-0.0042, 0.0710] |
| RC-DJB | 0.0562 | [0.0404, 0.0783] | 0.0154 | [-0.0288, 0.0559] |

### Paired Bootstrap Comparisons (vs. DJB)

| Baseline | MSE Diff. | 95% CI | Rank IC Diff. | 95% CI |
|----------|-----------|--------|---------------|--------|
| No-event | 0.01424 | [0.01145, 0.01765] | 0.0722 | [0.0204, 0.1293] |
| Concat fusion | 0.01140 | [0.00919, 0.01353] | 0.0323 | [-0.0214, 0.0875] |
| Event-jump SSM | 0.00136 | [0.00085, 0.00188] | 0.0076 | [-0.0192, 0.0370] |
| RC-DJB | 0.00097 | [0.00052, 0.00146] | 0.0176 | [-0.0174, 0.0557] |

### RC-DJB Intervention Tests

| Intervention | Event MSE Increase | 95% Paired CI |
|--------------|-------------------|---------------|
| No bridge | 0.0183 | [0.0165, 0.0203] |
| Zero text | 0.0037 | [0.0025, 0.0051] |
| Shuffled text | 0.0025 | [0.0014, 0.0036] |

DJB has the lowest held-out MSE with paired bootstrap improvements over every baseline. Rank IC is positive (0.0330) and stable under leave-one-ticker-out deletion (min 0.0263), but the ticker-clustered CI crosses zero — so the paper treats this as evidence about routing architecture, not a trading claim. The RC-DJB intervention tests confirm the bridge acts through volatility, volume, and uncertainty transport, not just return-mean shifting.

---

## Data

All data comes from public APIs — no proprietary, synthetic, or paid datasets.

| Source | Description |
|--------|-------------|
| [SEC EDGAR](https://www.sec.gov/edgar) | 8-K filings with `acceptanceDateTime` timestamps and primary document URLs |
| [Stooq](https://stooq.com) | Public daily OHLCV prices for 98 large-cap U.S. equities + SPY |
| [Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) | Optional Fama-French daily factors for FF3/FF4 label variants |

### Sample Summary

| Artifact | Count |
|----------|-------|
| Usable companies | 98 |
| SEC 8-K events | 7,236 |
| Matched no-event controls | 7,236 |
| Total feature rows | 14,472 |
| Public price rows | 159,093 |
| Held-out rows | 2,463 |

Generated data (raw filings, price CSVs, feature tensors, checkpoints, metrics) is stored locally under `data/` and is gitignored. See [`data/README.md`](data/README.md) for details.

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

# For text embeddings (sentence-transformers / BGE-small)
uv sync --extra embeddings
# or: pip install -e ".[embeddings]"

# For development tools (pytest, ruff)
uv sync --extra dev
# or: pip install -e ".[dev]"
```

### Environment Variables

```bash
cp .env.example .env
```

```env
# Required by SEC fair-access policy. Use your real contact information.
SEC_USER_AGENT="DJB academic research your-name your-email@example.com"

# Optional local controls
DJB_DATA_DIR="data"
DJB_DEVICE="auto"
```

---

## Reproduction

The full experiment is driven by numbered scripts in `scripts/`. Each script accepts `--config` and `--model-name` flags.

### Quick Smoke Test

```bash
python scripts/00_smoke.py --config configs/pilot.yaml
```

This runs a forward pass and loss computation on random data to verify the install.

### Full Paper Pipeline

```bash
# Step 1: Build SEC 8-K event rows from EDGAR
python scripts/10_build_events.py --config configs/paper_v3_bge.yaml

# Step 2: Download public daily prices
python scripts/20_download_prices.py --config configs/paper_v3_bge.yaml

# Step 3: Build leakage-safe feature tensors with BGE embeddings
python scripts/30_build_features.py --config configs/paper_v3_bge.yaml

# Step 4: Train a model (e.g., DJB)
python scripts/40_train.py --config configs/paper_v3_bge_djb.yaml --model-name djb

# Step 5: Evaluate on held-out split
python scripts/50_eval.py --config configs/paper_v3_bge_djb.yaml --model-name djb

# Step 6: Train and evaluate all baselines
python scripts/60_ablate.py --config configs/paper_v3_bge.yaml

# Step 7–9: Generate tables and figures for the paper
python scripts/70_make_tables.py --config configs/paper_v3_bge.yaml
python scripts/80_make_rc_djb_figures.py --config configs/paper_v3_bge.yaml
python scripts/85_make_paper_tables.py --config configs/paper_v3_bge.yaml

# Step 10: Replication sensitivity sweeps
python scripts/90_djb_replication_sweeps.py --config configs/paper_v3_bge.yaml
```

### Configurations

| Config | Description |
|--------|-------------|
| `configs/pilot.yaml` | Small-scale test (hashing embeddings, multi-form, 300 companies) |
| `configs/paper_v3_bge.yaml` | Paper-scale run (98 firms, BGE-small, 8-K only) |
| `configs/paper_v3_bge_djb.yaml` | DJB-specific overrides for paper run |
| `configs/paper_v3_bge_rc_djb.yaml` | RC-DJB-specific overrides |
| `configs/paper_v3_bge_ejssm.yaml` | Event-jump SSM baseline |
| `configs/paper_v3_bge_ff4.yaml` | Fama-French 4-factor label variant |
| `configs/experiments/` | Replication sweep configs |

---

## Project Structure

```
distributional-jump-bridges/
├── configs/                        # YAML experiment configurations
│   ├── model/                      #   Model architecture hyperparameters
│   └── experiments/                #   Replication sweep configs
├── data/                           # Generated data (gitignored)
│   ├── raw/                        #   Downloaded SEC filings and price CSVs
│   ├── processed/                  #   Feature bundles (.npz) and metadata
│   └── runs/                       #   Checkpoints, metrics, tables, figures
├── docs/                           # Paper manuscript and documentation
│   ├── main.tex                    #   LaTeX source
│   ├── djb.pdf                     #   Compiled paper
│   ├── references.bib              #   Bibliography (60+ references)
│   ├── figures/                    #   PDF figures used in manuscript
│   └── tables/                     #   Generated CSV result tables
├── scripts/                        # Numbered pipeline scripts (00–90)
│   ├── 00_smoke.py                 #   Code-path smoke test
│   ├── 10_build_events.py          #   SEC EDGAR event ingestion
│   ├── 20_download_prices.py       #   Public price download
│   ├── 30_build_features.py        #   Feature tensor construction
│   ├── 40_train.py                 #   Model training
│   ├── 50_eval.py                  #   Model evaluation
│   ├── 60_ablate.py                #   Baseline ablation sweep
│   ├── 70_make_tables.py           #   Result table generation
│   ├── 80_make_rc_djb_figures.py   #   Paper figure generation
│   ├── 85_make_paper_tables.py     #   LaTeX-ready table generation
│   └── 90_djb_replication_sweeps.py#   Sensitivity sweeps
├── src/cebt/                       # Python package
│   ├── models/cebt.py              #   All model architectures
│   ├── data/                       #   SEC client, price download, factors
│   │   ├── sec.py                  #     EDGAR API client
│   │   ├── prices.py               #     Price data fetching
│   │   └── factors.py              #     Fama-French factor loading
│   ├── features/                   #   Feature construction
│   │   ├── build.py                #     Leakage-safe feature builder
│   │   └── embeddings.py           #     Text embedding providers
│   ├── training/                   #   Training infrastructure
│   │   ├── train.py                #     Training loop
│   │   ├── losses.py               #     Composite loss function
│   │   └── dataset.py              #     PyTorch dataset
│   ├── evaluation/                 #   Evaluation and inference
│   │   ├── evaluate.py             #     Full evaluation pipeline
│   │   ├── metrics.py              #     Metric computation
│   │   ├── bootstrap.py            #     Bootstrap resampling
│   │   └── leakage.py              #     Leakage guard checks
│   ├── analysis/                   #   Result generation
│   │   ├── figures.py              #     Matplotlib figure code
│   │   └── tables.py               #     Table generation
│   ├── utils/                      #   Utilities
│   │   ├── config.py               #     YAML config loading
│   │   ├── io.py                   #     JSON/JSONL I/O
│   │   ├── time.py                 #     Trading calendar logic
│   │   └── hashing.py              #     Feature hashing
│   └── cli.py                      #   Shared CLI helpers
├── tests/                          # Test suite (10 test files)
├── pyproject.toml                  # Project metadata and dependencies
└── .env.example                    # Environment variable template
```

---

## Testing

```bash
# Run all tests
pytest

# Verbose output
pytest -v

# Specific test file
pytest tests/test_model_and_loss.py
```

### Test Coverage

| Area | Test Files | What It Verifies |
|------|-----------|-----------------|
| Timestamp discipline | `test_time_rules.py`, `test_evaluation_leakage.py` | Feature-label temporal ordering, after-hours label starts, leakage rejection |
| Feature construction | `test_feature_builder.py`, `test_factors.py` | Deterministic tensor generation, control matching, factor loading |
| Model correctness | `test_model_and_loss.py`, `test_training_smoke.py` | Finite losses for all 8 architectures, gradient flow, intervention behavior |
| Embeddings | `test_embeddings.py` | Hashing and sentence-transformer embedding providers |
| Evaluation | `test_bootstrap_and_tables.py` | Bootstrap resampling, table formatting, metric computation |
| Configuration | `test_config.py` | YAML loading, config resolution |

---

## Building the Paper

```bash
cd docs && tectonic main.tex
```

A pre-compiled snapshot is available at [`docs/djb.pdf`](docs/djb.pdf).

---

## Limitations

- **Sample scope**: 98 large-cap U.S. firms — not a broad equity universe. Generalization to small-cap, international, or low-liquidity samples is untested.
- **Data source**: Public daily prices (not CRSP or intraday). The return label is a conservative forward-drift target, not an immediate jump estimator.
- **Abnormal returns**: SPY-adjusted rather than full factor-model residuals (FF3/FF4 variants are available as configs).
- **Text encoder**: Frozen BGE-small with chunk mean pooling. Domain-specific encoders (FinBERT, BloombergGPT, etc.) could change absolute error levels — but the comparison isolates routing architecture, not encoder scale.
- **Rank IC**: Positive in point estimate and stable under LOTO deletion, but ticker-clustered CI crosses zero. This is not a trading-system claim.

---

## Responsible Use

This is a research project using public filings and public daily prices. The results **do not constitute trading recommendations**. The codebase is released to support leakage auditing, independent reproduction, and falsification.

---

## Author

[Aryan Gupta](https://github.com/aryan-cs)

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
