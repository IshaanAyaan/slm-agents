## Table 1 — Main results by condition

| Condition | n | Success rate | Cost/attempt (USD) | Cost/success (USD) | Escalation rate |
|---|---|---|---|---|---|
| C1 | 200 | 0.955 ± 0.015 | 0.00979 | 0.01025 | 0.000 |
| C2 | 200 | 0.635 ± 0.034 | 0.00286 | 0.00450 | 0.000 |
| C3 | 200 | 0.370 ± 0.034 | 0.00288 | 0.00778 | 0.030 |
| C4 | 200 | 0.710 ± 0.032 | 0.00188 | 0.00264 | 0.000 |
| C5 | 200 | 0.920 ± 0.019 | 0.00224 | 0.00244 | 0.000 |
| C6 | 200 | 0.810 ± 0.028 | 0.01842 | 0.02274 | 0.000 |

## Table 2 — Attribution decomposition

| Effect | Success rate | Cost/success (USD) |
|---|---|---|
| Harness effect, small model (C3−C2) | -0.265 | +0.00328 |
| Fine-tuning effect (C4−C2) | +0.075 | -0.00186 |
| Additive prediction for C5 (C3+C4−C2) | 0.445 | 0.00592 |
| Observed C5 | 0.920 | 0.00244 |
| Interaction (C5 − additive pred.) | +0.475 | -0.00349 |
| Super-additive? | True | True |
| Harness effect, large model (C6−C1) | -0.145 | +0.01248 |

## Table 3 — Token and reliability breakdown

| Condition | In-tok/attempt | Out-tok/attempt | Retry rate | Invalid-action rate | Verifier-failure rate | p50 latency (s) |
|---|---|---|---|---|---|---|
| C1 | 3518 | 63 | 0.00 | 0.00 | 0.00 | 23.41 |
| C2 | 6957 | 160 | 1.00 | 1.00 | 0.00 | 2.73 |
| C3 | 1999 | 181 | 2.01 | 2.54 | 2.56 | 2.38 |
| C4 | 4588 | 97 | 0.23 | 0.23 | 0.00 | 1.98 |
| C5 | 1977 | 129 | 0.04 | 0.06 | 0.06 | 2.08 |
| C6 | 2635 | 145 | 0.16 | 0.19 | 0.19 | 19.32 |

## Table 4 — Primary claim check (C5 vs C1)

| Quantity | Value |
|---|---|
| C1 cost/success (USD) | 0.01025 |
| C5 cost/success (USD) | 0.00244 |
| Cost reduction | 76.2% (target ≥ 50%) |
| Success drop | 3.5 pp (target ≤ 2 pp) |
| **Claim holds** | **False** |
