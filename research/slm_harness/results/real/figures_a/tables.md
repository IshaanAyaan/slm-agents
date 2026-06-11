## Table 1 — Main results by condition

| Condition | n | Success rate | Cost/attempt (USD) | Cost/success (USD) | Escalation rate |
|---|---|---|---|---|---|
| C1 | 80 | 0.725 ± 0.050 | 0.00132 | 0.00182 | 0.000 |
| C2 | 80 | 0.463 ± 0.056 | 0.00177 | 0.00382 | 0.000 |
| C3 | 80 | 0.263 ± 0.049 | 0.00056 | 0.00213 | 0.000 |
| C4 | 80 | 0.525 ± 0.056 | 0.00120 | 0.00229 | 0.000 |
| C5 | 80 | 0.975 ± 0.017 | 0.00044 | 0.00046 | 0.000 |
| C6 | 80 | 0.550 ± 0.056 | 0.00207 | 0.00376 | 0.025 |

## Table 2 — Attribution decomposition

| Effect | Success rate | Cost/success (USD) |
|---|---|---|
| Harness effect, small model (C3−C2) | -0.200 | -0.00169 |
| Fine-tuning effect (C4−C2) | +0.062 | -0.00153 |
| Additive prediction for C5 (C3+C4−C2) | 0.325 | 0.00060 |
| Observed C5 | 0.975 | 0.00046 |
| Interaction (C5 − additive pred.) | +0.650 | -0.00015 |
| Super-additive? | True | True |
| Harness effect, large model (C6−C1) | -0.175 | +0.00194 |

## Table 3 — Token and reliability breakdown

| Condition | In-tok/attempt | Out-tok/attempt | Retry rate | Invalid-action rate | Verifier-failure rate | p50 latency (s) |
|---|---|---|---|---|---|---|
| C1 | 3074 | 132 | 0.05 | 0.05 | 0.00 | 10.65 |
| C2 | 11063 | 489 | 0.23 | 0.23 | 0.00 | 9.15 |
| C3 | 1729 | 162 | 1.74 | 2.48 | 2.48 | 2.95 |
| C4 | 7134 | 337 | 0.45 | 0.45 | 0.00 | 3.41 |
| C5 | 1379 | 130 | 0.03 | 0.03 | 0.03 | 3.00 |
| C6 | 3950 | 210 | 1.65 | 1.93 | 1.93 | 17.99 |

## Table 4 — Primary claim check (C5 vs C1)

| Quantity | Value |
|---|---|
| C1 cost/success (USD) | 0.00182 |
| C5 cost/success (USD) | 0.00046 |
| Cost reduction | 74.9% (target ≥ 50%) |
| Success drop | -25.0 pp (target ≤ 2 pp) |
| **Claim holds** | **True** |
