# Research Plan

1. Build a leakage-safe SEC 8-K event panel with accepted timestamps, public daily prices, matched
   no-event controls, and aligned firm/market return windows.
2. Train no-text, text-only, concatenation, bottleneck, operator, state-jump, DJB, and RC-DJB
   variants under the same chronological split and frozen text representation.
3. Evaluate event-response regression, probabilistic calibration, Rank IC, intervention response,
   per-channel transport, and label-window correctness.
4. Use RC-DJB as a routing diagnostic rather than the headline forecaster: it tests what changes
   when disclosure text cannot directly move the abnormal-return mean.
5. Stress-test the result with temperature sensitivity, leave-one-ticker-out analysis, item-type
   diagnostics, leakage guards, and corrected firm/market return-window alignment.

## Current Findings

- The current paper-scale run uses 7,236 real SEC 8-K events, 7,236 matched no-event controls, and
  2,463 held-out rows across 98 firms.
- DJB has the best corrected multi-target MSE among the main frozen-embedding routing models.
- RC-DJB provides the cleanest routing audit: return-mean movement is blocked by construction, while
  volatility, volume, and uncertainty channels remain measurable.
- The Rank IC point estimate is positive but not ticker-cluster significant, so the paper frames it
  as evidence about architectural routing rather than a trading-system claim.
- Generated data, synthetic labels, paid data, and closed-source model outputs are not used as
  evidence for the reported results.
