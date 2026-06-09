# H100 Runbook — running the real C1–C6 experiment end to end (zero API)

This runs the entire study on your own H100s with **no paid API tokens**: a strong
**open** model serves as the large/teacher model, two small models (Qwen3-4B, Qwen3-1.7B)
are the students, and "cost" is computed from measured GPU throughput. One model family
is resident at a time, so it works even on a modest GPU count (70B teacher needs ~2×H100;
students fit on 1).

## 0. Credentials — do this safely (not in the repo)

The VPN/SSH credentials must **never** go in this git repo (it pushes to GitHub). Store
them in the macOS Keychain on your Mac and rotate the ones that were shared in chat:

```bash
# on your Mac — store the VPN password + PSK in the login Keychain
security add-generic-password -a "USER70" -s "masason-vpn-password" -w
security add-generic-password -a "USER70" -s "masason-vpn-psk" -w
# retrieve when needed:
security find-generic-password -a "USER70" -s "masason-vpn-password" -w
```

Then connect the VPN with macOS's built-in L2TP/IPsec client (System Settings → Network →
VPN), using URL `masason-foundation.vsr.variosecure.net` / server `172.16.255.70`, user
`USER70`. **Rotate the password and PSK** afterward, since they appeared in chat.

> The agent cannot reach this VPN or your H100s from its sandbox and does not store these
> credentials anywhere. You drive the box; the agent provides the code and analysis.

## 1. Get the box ready

```bash
# SSH into the H100 box (through the VPN, from your Mac)
git clone https://github.com/LuxenAI/slm-harness.git && cd slm-harness
git checkout fable
cp research/slm_harness/infra/config.env.example research/slm_harness/infra/config.env
$EDITOR research/slm_harness/infra/config.env     # set GPU_HOURLY_USD, models, repos
huggingface-cli login                              # for gated weights (Llama); Qwen is open
```

`config.env` and everything under `results/real/` and `training/checkpoints/` are
gitignored — nothing sensitive or large is ever committed.

## 2. Run it (one command)

```bash
bash research/slm_harness/infra/run_all.sh
```

Stages (each resumable: `run_all.sh <stage>`):

| stage | what it does | resident model | rough time* |
|---|---|---|---|
| `preflight` | check GPUs, disk, deps | — | seconds |
| `setup` | pip install vllm/peft/trl + run unit tests | — | ~10 min |
| `genbench` | generate real nav tasks over your repos (deterministic ground truth) | — | seconds |
| `teacher` | serve open 70B → measure price → gen C1/C6 train trajectories → distill → eval C1,C6 on test | teacher | 1–3 h |
| `train` | LoRA-tune Qwen3-4B (+1.7B) on the distilled data | — (training) | 20–60 min |
| `students_4b` | serve 4B base+LoRA → measure price → eval C2,C3,C4,C5 on test | 4B | 20–40 min |
| `students_17b` | same for the 1.7B ablation | 1.7B | 15–30 min |
| `report` | merge runs, compute metrics + attribution, figures, fill the white paper | — | seconds |

\* depends on GPU count, teacher size, task counts.

## 3. Outputs

```
research/slm_harness/results/real/
  metrics_4b.json     headline: success, cost/success, 2x3 attribution, claim, break-even
  metrics_17b.json    1.7B ablation
  figures_4b/         fig1–5 + tables.md (NOT watermarked — real runs via --real)
  runs_4b.jsonl       every (condition,task,seed) trajectory with tokens/cost/verifier
research/slm_harness/paper/white_paper.md   auto-filled with the real numbers
```

## 4. Cost model (why this is API-free and fair)

Each served model's price comes from `cost_model.py`, which benchmarks the live endpoint
and converts measured prefill/decode tokens-per-second + your `GPU_HOURLY_USD` × GPUs into
per-token USD. The 70B teacher is correctly costlier per token (more GPUs, fewer tok/s);
the students are cheap. Fine-tuning cost = measured training wall-time × GPUs × rate, fed
into the break-even analysis. No external price list, no API bill.

## 5. Scaling up / SWE-bench later

Add more `BENCH_REPOS` (any Python repo path:split) for more tasks and stronger
generalization. To extend beyond navigation to SWE-bench-Verified-style tasks, write an
adapter that materializes each instance's repo checkout as a `workspace` and emits the
same task schema; the runner/harness/metrics are unchanged.

## 6. If something stalls

- vLLM logs: `results/real/logs/{teacher,students_*}.log`.
- Re-run a single stage, e.g. `bash research/slm_harness/infra/run_all.sh students_4b`.
- Out-of-memory on the teacher: lower `--max-model-len` in `serve_vllm.sh` or raise
  `TEACHER_TP`. Use Qwen2.5-72B (open, ungated) if Llama access is pending.
