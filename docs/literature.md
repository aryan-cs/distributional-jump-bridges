# Literature Map

This repository supports the paper on Distributional Jump Bridges (DJB) for SEC disclosure
response modeling. The dataset is an experimental substrate; the main contribution is the
distributional bridge architecture and its routing diagnostics.

## Nearest Work

- Event studies and disclosure-response econometrics establish the empirical setting for measuring
  post-disclosure return, volatility, and volume responses.
- Financial text and disclosure-forecasting models provide the motivation for conditioning market
  response models on public filings.
- Counterfactual, causal, and leakage-aware time-series models motivate the strict timestamp
  discipline used here, but DJB makes the event effect a learned transport from a no-event
  predictive distribution to an event-response distribution.
- Time-series foundation models and financial language models are relevant encoders/backbones, but
  this draft isolates the routing architecture with controlled frozen text embeddings.

## Positioning

DJB differs from standard fusion by predicting a no-event distribution first and then applying a
disclosure-conditioned jump in response space. The return-conservative variant is a diagnostic
constraint: it blocks direct abnormal-return mean movement while leaving risk, liquidity, and
uncertainty transport available.
