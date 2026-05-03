# Research Plan

1. Verify that a no-event dynamics model can learn ordinary short-horizon movement from pre-event windows.
2. Add disclosure embeddings and show naive concatenation can overfit or fail to isolate event impact.
3. Train CEBT with a stochastic event bottleneck and matched controls.
4. Evaluate whether true disclosures produce larger event deltas than pseudo-events.
5. Ablate bottleneck size, control loss, and event text to test whether the architectural constraints matter.
6. Scale from 8-K-only pilot to 8-K/10-Q/10-K main runs after leakage checks pass.

## Current Paper-Scale V1 Findings

- Built 1,256 real SEC 8-K events across 20 companies from 2021-2025.
- Matched one same-ticker no-event control per filing, yielding 2,512 feature rows.
- CEBT variants improve held-out MSE relative to no-event, text-only, and concat-fusion baselines.
- Full CEBT reduces matched-control residual magnitude from 0.0494 in the no-control variant to 0.0164.
- Rank IC confidence intervals still cross zero, so the current result supports architectural decomposition more than trading-signal strength.
