# Specialized Small-Model Subagents

**Co-designing a fine-tuned small language model and a task-specific harness for cost-efficient multi-agent systems.**

Ishaan Ranjan and Ganesh Talluri

📄 **Paper**: [`paper/white_paper.pdf`](paper/white_paper.pdf) · Raw data, metrics, figures, and SFT sets under [`slm_harness/results/real/`](slm_harness/results/real/)

---

## The result in one table

Multi-agent systems delegate sub-tasks to worker models. Swapping the expensive worker for a cheap general one is the standard cost fix, and it quietly fails. We instead fine-tune a small model for one narrow role and pair it with a harness built for that role, and we measure everything by cost per successful task on real hardware.

Full 2×3 factorial, one NVIDIA H100 NVL, Qwen2.5-72B-Instruct-AWQ as teacher and baseline, 200 held-out test tasks across five real repositories (flask, click, rich, httpx, plus one private corpus), zero paid API calls. Per-task statistics, n = 200 per condition, 3B student grid:

| Condition | Success [95% CI] | Cost per success |
|---|---|---|
| C1 — 72B, generic harness (baseline) | 0.955 [0.917, 0.976] | $0.010252 |
| C2 — 3B, generic harness (naive swap) | 0.480 [0.412, 0.549] | $0.013908 |
| C3 — 3B, custom harness | 0.250 [0.195, 0.314] | $0.010199 |
| C4 — 3B fine-tuned, generic harness | 0.410 [0.344, 0.479] | $0.009759 |
| **C5 — 3B fine-tuned, custom harness (ours)** | **0.945 [0.904, 0.969]** | **$0.001558** |
| C6 — 72B, custom harness | 0.810 [0.750, 0.858] | $0.022736 |

**The 3B specialist matches the 72B baseline on success (McNemar p = 0.82) at 84.8% lower cost per successful task (bootstrap 95% CI [83.3, 86.1]).** The naive swap (C2) is significantly worse than the baseline on success *and* on cost at the same time. Each ingredient alone hurts (C3, C4 both underperform C2); only the co-designed combination works. At 1.5B the contrast is starkest, with the same fine-tuned model scoring 1.0% in the generic harness and 93.5% in the custom one.

The result holds on every one of the five test repositories, across a 1.5B–7B size sweep (76–85% cost reductions, all statistically indistinguishable from the baseline on success), and replicates an earlier independent run at smaller scale.

## Phase 2 — parameter floors for developer subroutines (sub-500M)

Phase 1 showed a 1.5B–3B specialist can carry a whole subagent role. Phase 2 asks how much further the recipe compresses: we decompose coding-agent work into **8 narrow, deterministically-verifiable subroutines** (JSON action repair, action routing, path normalization, search-query generation, search-hit ranking, read-span selection, evidence judging, traceback localization), generate **oracle-labeled data from real repos with no teacher as judge** (train: flask/click/rich; held-out test: httpx/jinja2, plus held-out phrasing templates), and fully fine-tune specialists at **135M / 360M / 0.5B / 1.5B**. Every subroutine also gets a rules-only baseline, and where rules already win we say so.

📄 **Phase-2 paper**: [`paper/phase2/main.pdf`](paper/phase2/main.pdf) · Parameter-floor map, per-cell results, and figures under [`slm_harness/results/subroutines/`](slm_harness/results/subroutines/)

```bash
# offline: regenerate all Phase-2 datasets deterministically (clones pinned repo tags)
python -m slm_harness.scripts.gen_subroutine_data --out /tmp/p2 --repos-cache /tmp/repos
# GPU pod: full pipeline (datagen -> SFT -> 4-size sweep -> eval -> parameter floor)
bash slm_harness/infra/run_phase2.sh
# MCP proof of concept (rules backend, no GPU needed)
python -m slm_harness.mcp.server
```

## Repo map

```
paper/                  LaTeX white paper + compiled PDF
slm_harness/            the study
  harness/              generic baseline harness + the custom co-designed harness
  verifier.py           deterministic per-step and final-answer verification
  distill.py            teacher trajectories -> SFT data (the co-design step)
  runner.py             (condition x task) grid runner with cost accounting
  metrics.py            attribution decomposition, claim check, break-even
  analysis.py           per-task dedup, Wilson CIs, exact McNemar, paired bootstrap
  tasks/realbench/      benchmark generator over real repos (AST-grounded, no labeling)
  training/             LoRA fine-tuning + adapter merge
  infra/                turnkey single-GPU pipeline (run_all.sh) + vLLM serving + cost model
  results/real/         all run logs, metrics, significance, figures, and SFT data
  tests/                offline test suite (no GPU or network needed)
openharness/            small vendored MIT tool layer the harnesses run on (see NOTICE.md)
```

## Reproduce

Offline checks (any machine, no GPU):

```bash
pip install -e ".[dev]"
pytest slm_harness/tests -q                      # 59 tests
python slm_harness/scripts/run_smoke.py          # synthetic end-to-end smoke run
python slm_harness/scripts/compute_significance.py \
  --runs slm_harness/results/real/eval_teacher/runs.jsonl \
         slm_harness/results/real/eval_b_base/runs.jsonl \
         slm_harness/results/real/eval_b_ft/runs.jsonl    # recompute paper stats from frozen logs
```

Full experiment (one 80GB+ GPU, ~1.5 GPU-hours, ~$5):

```bash
cp slm_harness/infra/config.runpod.env.example slm_harness/infra/config.env
# edit GPU_HOURLY_USD and models if needed
bash slm_harness/infra/run_all.sh
```

The pipeline stages are preflight, venv setup, benchmark generation, teacher serving + distillation + C1/C6 eval, LoRA training for each student, C2–C5 eval per student, and the report. Each stage is resumable by name, e.g. `bash slm_harness/infra/run_all.sh students_b`.

## Method in one paragraph

A fixed orchestrator delegates file-navigation tasks to the subagent under test. The custom harness exposes four JSON actions (SEARCH, READ, ANSWER, ESCALATE), keeps all state outside the model, validates every action against the current state machine before execution, and re-prompts only the failed step. The large model runs the training split inside both harnesses; verifier-clean trajectories become SFT data; each student is LoRA fine-tuned on that data and served merged. The 2×3 grid over {large, small, small fine-tuned} × {generic, custom} separates the harness effect, the fine-tuning effect, and their interaction. Costs come from measured tokens/sec on the same GPU at a fixed hourly rate, so the headline ratios are independent of the rate chosen.

## License

MIT. The `openharness/` directory is a vendored subset of the MIT-licensed OpenHarness project (attribution in [NOTICE.md](NOTICE.md)).
