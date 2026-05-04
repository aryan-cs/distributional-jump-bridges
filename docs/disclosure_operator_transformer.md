# Disclosure Operator Baseline

This note documents a low-rank operator baseline used for controlled comparison with DJB.

## Hypothesis

If SEC disclosure text contains event-specific response information, then its effect may be
represented as a low-rank transformation of the pre-event latent market state. The transformation
should improve prediction while remaining sparse enough to inspect.

## Architecture

1. A no-event encoder maps pre-event prices and volumes to a latent state.
2. Disclosure text and event metadata generate a sample-specific low-rank operator.
3. The operator perturbs the latent state before the outcome head predicts abnormal return,
   volatility jump, and volume jump.
4. Matched controls keep the operator close to identity when no disclosure is present.

The paper uses this baseline to test whether distributional response transport is more effective
than latent-state operator fusion under the same timestamps, labels, and text embeddings.
