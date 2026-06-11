## Table 1 — Main results by condition

| Condition | n | Success rate | Cost/attempt (USD) | Cost/success (USD) | Escalation rate |
|---|---|---|---|---|---|
| C1 | 200 | 0.955 ± 0.015 | 0.00979 | 0.01025 | 0.000 |
| C2 | 200 | 0.325 ± 0.033 | 0.00108 | 0.00332 | 0.000 |
| C3 | 200 | 0.000 ± 0.000 | 0.00221 | — | 0.000 |
| C4 | 200 | 0.010 ± 0.007 | 0.00129 | 0.12899 | 0.000 |
| C5 | 200 | 0.935 ± 0.017 | 0.00167 | 0.00179 | 0.010 |
| C6 | 200 | 0.810 ± 0.028 | 0.01842 | 0.02274 | 0.000 |

## Table 2 — Attribution decomposition

| Effect | Success rate | Cost/success (USD) |
|---|---|---|
| Harness effect, small model (C3−C2) | -0.325 | +inf |
| Fine-tuning effect (C4−C2) | -0.315 | +0.12568 |
| Additive prediction for C5 (C3+C4−C2) | -0.315 | inf |
| Observed C5 | 0.935 | 0.00179 |
| Interaction (C5 − additive pred.) | +1.250 | -inf |
| Super-additive? | True | True |
| Harness effect, large model (C6−C1) | -0.145 | +0.01248 |

## Table 3 — Token and reliability breakdown

| Condition | In-tok/attempt | Out-tok/attempt | Retry rate | Invalid-action rate | Verifier-failure rate | p50 latency (s) |
|---|---|---|---|---|---|---|
| C1 | 3518 | 63 | 0.00 | 0.00 | 0.00 | 23.41 |
| C2 | 3435 | 84 | 0.01 | 0.01 | 0.00 | 0.99 |
| C3 | 1223 | 187 | 2.03 | 3.03 | 3.03 | 1.14 |
| C4 | 2663 | 101 | 0.49 | 0.49 | 0.00 | 0.62 |
| C5 | 1888 | 137 | 0.20 | 0.23 | 0.23 | 1.31 |
| C6 | 2635 | 145 | 0.16 | 0.19 | 0.19 | 19.32 |

## Table 4 — Primary claim check (C5 vs C1)

| Quantity | Value |
|---|---|
| C1 cost/success (USD) | 0.01025 |
| C5 cost/success (USD) | 0.00179 |
| Cost reduction | 82.6% (target ≥ 50%) |
| Success drop | 2.0 pp (target ≤ 2 pp) |
| **Claim holds** | **True** |
