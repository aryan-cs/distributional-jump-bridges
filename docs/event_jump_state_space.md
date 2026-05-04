# State-Jump Baseline

This note documents a latent state-jump baseline used for comparison with Distributional Jump
Bridges. It is not the paper's main contribution.

## Hypothesis

Financial disclosures can be represented as sparse jumps in latent market dynamics. If this
hypothesis is useful, a model with explicit disclosure jumps should improve event-window forecasting
and produce larger jump magnitudes for real disclosures than for matched no-event controls.

## Implementation

1. A GRU state encoder maps 40 pre-event trading days to a latent state.
2. A no-event transition predicts the ordinary continuation.
3. Disclosure text generates a bounded jump vector.
4. The jumped state predicts abnormal return, volatility jump, and volume jump.
5. A Gaussian head provides a calibration target through negative log likelihood.

The baseline is intentionally trained on the same SEC 8-K feature bundle as DJB so that differences
come from routing architecture rather than data construction.
