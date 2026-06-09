## Table 1 — Main results by condition

> **NOTE: numbers below were produced with STUB model clients and are synthetic pipeline-validation output, not research results.**

| Condition | n | Success rate | Cost/attempt (USD) | Cost/success (USD) | Escalation rate |
|---|---|---|---|---|---|
| C1 | 50 | 1.000 ± 0.000 | 0.01530 | 0.01530 | 0.000 |
| C2 | 50 | 0.200 ± 0.057 | 0.00050 | 0.00248 | 0.000 |
| C3 | 50 | 0.900 ± 0.042 | 0.00016 | 0.00018 | 0.000 |
| C4 | 50 | 0.680 ± 0.066 | 0.00050 | 0.00074 | 0.000 |
| C5 | 50 | 0.900 ± 0.042 | 0.00005 | 0.00006 | 0.000 |
| C6 | 50 | 0.900 ± 0.042 | 0.00371 | 0.00413 | 0.000 |

## Table 2 — Attribution decomposition

| Effect | Success rate | Cost/success (USD) |
|---|---|---|
| Harness effect, small model (C3−C2) | +0.700 | -0.00230 |
| Fine-tuning effect (C4−C2) | +0.480 | -0.00174 |
| Additive prediction for C5 (C3+C4−C2) | 1.380 | -0.00156 |
| Observed C5 | 0.900 | 0.00006 |
| Interaction (C5 − additive pred.) | -0.480 | +0.00162 |
| Super-additive? | False | False |
| Harness effect, large model (C6−C1) | -0.100 | -0.01117 |

## Table 3 — Token and reliability breakdown

| Condition | In-tok/attempt | Out-tok/attempt | Retry rate | Invalid-action rate | Verifier-failure rate | p50 latency (s) |
|---|---|---|---|---|---|---|
| C1 | 4849 | 50 | 0.00 | 0.00 | 0.00 | 0.01 |
| C2 | 4832 | 42 | 1.20 | 1.20 | 0.00 | 0.01 |
| C3 | 1306 | 108 | 1.60 | 1.60 | 1.60 | 0.01 |
| C4 | 9854 | 83 | 0.40 | 0.40 | 0.00 | 0.01 |
| C5 | 919 | 88 | 0.40 | 0.40 | 0.40 | 0.01 |
| C6 | 835 | 81 | 0.00 | 0.00 | 0.00 | 0.01 |

## Table 4 — Primary claim check (C5 vs C1)

| Quantity | Value |
|---|---|
| C1 cost/success (USD) | 0.01530 |
| C5 cost/success (USD) | 0.00006 |
| Cost reduction | 99.6% (target ≥ 50%) |
| Success drop | 10.0 pp (target ≤ 2 pp) |
| **Claim holds** | **False** |
