# Research Plan

1. Verify that a no-event dynamics model can learn ordinary short-horizon movement from pre-event windows.
2. Add disclosure embeddings and show naive concatenation can overfit or fail to isolate event impact.
3. Train CEBT with a stochastic event bottleneck and matched controls.
4. Evaluate whether true disclosures produce larger event deltas than pseudo-events.
5. Ablate bottleneck size, control loss, and event text to test whether the architectural constraints matter.
6. Extend beyond the current 8-K-only paper run to 10-Q/10-K experiments after leakage checks pass.

## Current Paper-Scale V2 Findings

- Built 2,971 real SEC 8-K events across 50 companies from 2021-2025.
- Matched one same-ticker no-event control per filing, yielding 5,942 feature rows.
- Full CEBT has positive abnormal-return rank IC with a bootstrap interval above zero: 0.0623 [0.0042, 0.1222].
- Full CEBT significantly improves paired MSE versus the same architecture without control losses.
- Full CEBT reduces matched-control residual magnitude from 0.1095 in the no-control variant to 0.0199.
- Concat fusion has slightly lower point-estimate MSE than full CEBT, but the paired MSE difference is not statistically resolved.
