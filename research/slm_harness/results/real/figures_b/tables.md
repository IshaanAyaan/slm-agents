## Table 1 — Main results by condition

| Condition | n | Success rate | Cost/attempt (USD) | Cost/success (USD) | Escalation rate |
|---|---|---|---|---|---|
| C1 | 200 | 0.955 ± 0.015 | 0.00979 | 0.01025 | 0.000 |
| C2 | 200 | 0.480 ± 0.035 | 0.00668 | 0.01391 | 0.000 |
| C3 | 200 | 0.250 ± 0.031 | 0.00255 | 0.01020 | 0.005 |
| C4 | 200 | 0.410 ± 0.035 | 0.00400 | 0.00976 | 0.000 |
| C5 | 200 | 0.945 ± 0.016 | 0.00147 | 0.00156 | 0.000 |
| C6 | 200 | 0.810 ± 0.028 | 0.01842 | 0.02274 | 0.000 |

## Table 2 — Attribution decomposition

| Effect | Success rate | Cost/success (USD) |
|---|---|---|
| Harness effect, small model (C3−C2) | -0.230 | -0.00371 |
| Fine-tuning effect (C4−C2) | -0.070 | -0.00415 |
| Additive prediction for C5 (C3+C4−C2) | 0.180 | 0.00605 |
| Observed C5 | 0.945 | 0.00156 |
| Interaction (C5 − additive pred.) | +0.765 | -0.00449 |
| Super-additive? | True | True |
| Harness effect, large model (C6−C1) | -0.145 | +0.01248 |

## Table 3 — Token and reliability breakdown

| Condition | In-tok/attempt | Out-tok/attempt | Retry rate | Invalid-action rate | Verifier-failure rate | p50 latency (s) |
|---|---|---|---|---|---|---|
| C1 | 3518 | 63 | 0.00 | 0.00 | 0.00 | 23.41 |
| C2 | 9254 | 425 | 0.40 | 0.40 | 0.00 | 3.54 |
| C3 | 1933 | 167 | 2.01 | 2.75 | 2.75 | 1.75 |
| C4 | 6493 | 337 | 0.35 | 0.35 | 0.00 | 2.08 |
| C5 | 1833 | 126 | 0.03 | 0.03 | 0.03 | 1.69 |
| C6 | 2635 | 145 | 0.16 | 0.19 | 0.19 | 19.32 |

## Table 4 — Primary claim check (C5 vs C1)

| Quantity | Value |
|---|---|
| C1 cost/success (USD) | 0.01025 |
| C5 cost/success (USD) | 0.00156 |
| Cost reduction | 84.8% (target ≥ 50%) |
| Success drop | 1.0 pp (target ≤ 2 pp) |
| **Claim holds** | **True** |
