# Research Plan

1. Verify that a no-event dynamics model can learn ordinary short-horizon movement from pre-event windows.
2. Add disclosure embeddings and show naive concatenation can overfit or fail to isolate event impact.
3. Train CEBT with a stochastic event bottleneck and matched controls.
4. Evaluate whether true disclosures produce larger event deltas than pseudo-events.
5. Ablate bottleneck size, control loss, and event text to test whether the architectural constraints matter.
6. Extend beyond the current 8-K-only paper run to 10-Q/10-K experiments after leakage checks pass.

## Current Paper-Scale V3 Findings

- Built 7,236 real SEC 8-K events across 98 usable companies from 2020-2025.
- Matched one same-ticker no-event control per filing, yielding 14,472 feature rows.
- Full CEBT has positive abnormal-return rank IC under iid, ticker-clustered, and month-clustered intervals: 0.0463 with ticker-clustered 95% CI [0.0052, 0.0893].
- Full CEBT significantly improves paired rank IC versus concat fusion and no-event dynamics.
- Full CEBT significantly improves paired MSE versus no-event and CEBT without controls, but concat fusion has better point-estimate and paired MSE.
- Full CEBT reduces matched-control residual magnitude from 0.0298 in the no-control variant to 0.0103.
