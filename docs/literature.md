# Literature Map

CEBT is an architecture paper. The dataset is only a substrate for testing whether an event bottleneck improves leakage-safe event modeling.

## Nearest Work

- CAMEF: causal-augmented multi-modal event-driven financial forecasting.
- Causal Transformer and G-Transformer: counterfactual outcome estimation over temporal data.
- Chronos, Time-LLM, MOMENT, Timer-XL: time-series foundation models and LLM reprogramming.
- Profit Mirage / FinLake-Bench: leakage in LLM-based financial agents.
- Subliminal Learning: hidden trait transfer through generated data, relevant to generated financial summaries and distillation.

## Positioning

CEBT differs by decomposing predictions into no-event dynamics and event-induced residuals, with the residual forced through a compact stochastic event bottleneck and trained against no-event controls.
