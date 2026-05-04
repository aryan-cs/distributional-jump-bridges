# Event-Jump State Space Model

The Event-Jump State Space Model (EJSSM) reframes disclosure modeling as a latent dynamics
problem. Ordinary pre-event market behavior evolves through a no-event state transition. A public
disclosure then introduces a learned jump in that latent state.

## Hypothesis

Financial disclosures are not merely text covariates. They are timestamped interventions that
create sparse discontinuities in latent market dynamics. If the hypothesis is correct, a model with
explicit disclosure jumps should improve event-window forecasting and produce jump magnitudes that
separate real disclosures from matched no-event controls.

## Minimal Test In This Repository

The first implementation fits the existing tensor pipeline:

1. A GRU state encoder maps 40 pre-event trading days to a latent state.
2. A no-event transition predicts the ordinary continuation.
3. Disclosure text and metadata generate a bounded jump vector.
4. The jumped state predicts abnormal return, volatility jump, and volume jump.
5. A Gaussian head provides a first calibration target through a negative log-likelihood loss.

The model is intentionally comparable to existing baselines on the same SEC 8-K feature bundle.
It should not be promoted as a paper result unless it beats strong fusion baselines under clustered
uncertainty and shows event-vs-control jump separation.
