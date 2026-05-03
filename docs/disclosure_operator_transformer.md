# Disclosure Operator Transformer

The pivot architecture treats a disclosure as a state transition operator rather than as an
embedding to concatenate with a price forecaster.

## Hypothesis

If SEC disclosure text contains event-specific market information, then its effect should be
representable as a low-rank transformation of the pre-event latent market state. The resulting
operator should be useful for prediction, sparse enough to inspect, and transplantable to matched
no-event states as a controlled intervention.

## Architecture

1. A no-event Transformer encoder maps pre-event prices and volumes to a latent state.
2. Disclosure text and event metadata generate a sample-specific low-rank operator.
3. The operator perturbs the latent state before the outcome head predicts abnormal return,
   volatility jump, and volume jump.
4. Matched controls keep the operator close to identity when no disclosure is present.

## Scientific Claim

The contribution is not another event dataset. The scientific claim is that event-driven financial
forecasting can be framed as **learned state surgery**: public information acts as an operator on
market beliefs. This can be tested by prediction, operator norm diagnostics, operator
transplantation, and composition experiments.

## Closest Related Work

- time2time performs post-hoc hidden-state activation transplantation in time-series foundation
  models.
- Koopman forecasting methods learn global or local latent dynamics operators.
- CAMEF and event-aware forecasting fuse text/events with time-series states.

DOT differs by learning disclosure-conditioned low-rank operators directly from event text and
training them with no-event placebo controls.
