# SLM + co-designed harness as a subagent: research scaffold

## Status: real run completed (2026-06-10) — primary claim holds

The full 2×3 factorial ran end-to-end on one self-hosted NVIDIA A40 (Qwen2.5-14B-Instruct
teacher, Qwen2.5-3B-Instruct student, vLLM 0.7.3, zero API spend). Deduplicated per-task
statistics over the 40 held-out test tasks (`results/real/significance.json`):

| Condition | Success [95% CI] | Cost/success |
|---|---|---|
| C1 large+generic (baseline) | 0.725 [0.572, 0.839] | $0.001818 |
| C2 small+generic (naive swap) | 0.475 [0.329, 0.625] | $0.003715 — **2.0× worse than C1** |
| C3 small+custom | 0.275 [0.161, 0.428] | $0.002041 |
| C4 fine-tuned+generic | 0.525 [0.375, 0.671] | $0.002291 |
| **C5 fine-tuned+custom (proposed)** | **0.975 [0.871, 0.996]** | **$0.000457** |
| C6 large+custom | 0.550 [0.398, 0.693] | $0.003762 |

C5 vs C1: **−74.9%** cost-per-success (bootstrap 95% CI [63.8%, 84.0%]), **+25 pp**
success (exact McNemar p = 0.006). Interaction on success is super-additive (+0.650 over
the additive prediction). Full write-up: [`paper/white_paper.md`](paper/white_paper.md);
raw trajectories and metrics are committed under `results/real/`. To reproduce on a
RunPod-style pod: copy `infra/config.runpod.env.example` to `infra/config.env`, set your
GPU rate, and run `bash research/slm_harness/infra/run_all.sh` (~1 GPU-hour on an A40).

## Research question

Multi-agent systems typically pair a frontier-model orchestrator with cheaper
*general-purpose* worker models. Those workers stay bloated — large prompts, full tool
lists, free-form context, reasoning overhead — and their failures cause retries and
orchestrator corrections that eat the per-token savings.

**Hypothesis:** a small language model fine-tuned for ONE subagent role, paired with a
co-designed task-specific harness (which owns state, memory, valid actions,
verification, and retry localization), beats a cheap general-purpose subagent on
**cost-per-successful-task** while preserving task success rate.

**Primary claim (C5 vs C1):** ≥ 50% reduction in cost-per-successful-task with ≤ 2 pp
drop in success rate.

**Phase-1 subagent role:** file/code search & navigation — high-volume, low-reasoning,
naturally scoped, and directly served by existing OpenHarness tools (`grep`,
`read_file`, `glob`).

## The 2×3 factorial design (the methodological core)

|  | generic harness | custom harness |
|---|---|---|
| **large general model** | **C1** baseline | **C6** upper bound |
| **small general model** | **C2** naive cost-cut | **C3** harness-only effect |
| **small fine-tuned model** | **C4** fine-tuning-only effect | **C5** proposed system |

Attribution decomposition (in `metrics.py`):
harness effect = C3 − C2; fine-tuning effect = C4 − C2;
additive prediction for C5 = C3 + C4 − C2;
**interaction = C5 − (C3 + C4 − C2)** — super-additive iff positive on benefit metrics
(negative on cost metrics). Plus the large-model harness effect C6 − C1.

## Layout

See `REPO_MAP.md` for how this connects to the existing OpenHarness/ohmo codebase
(what is reused, what is deliberately not, what must not be touched).

```
schemas/        config (models/harnesses/conditions/pricing/run), tasks, actions, run logs
conditions.py   default C1–C6 registry (placeholder model names/prices, clearly labeled)
model_clients/  protocol + Anthropic adapter + OpenAI-compatible (vLLM) adapter + stubs
harness/        generic.py (baseline), custom.py (narrow harness), state.py (external state)
verifier.py     deterministic step verifier + shared final task scorer
runner.py       (condition × task × seed) grid runner, JSONL logging, cost accounting
metrics.py      all metrics + attribution + super-additivity + claim check + break-even
distill.py      successful C1/C6 trajectories → SFT JSONL (train/val/test)
figures.py      figures 1–5 + numbered tables (markdown)
tasks/smoke/    offline smoke benchmark (synthetic fixture repo, 10 tasks)
training/       LoRA training interface + README (NOT auto-executed)
scripts/        run_smoke / compute_metrics / build_distillation / make_figures
tests/          scaffold tests (separate from the main repo suite)
```

## Quick start

Everything below is **offline** — stub model clients, no credentials, no network.

```bash
# from the repo root
pip install -e ".[dev]"          # or at minimum: pydantic pytest pytest-asyncio anthropic openai matplotlib

# 1. tests
pytest research/slm_harness/tests/ -q          # scaffold tests (44)
pytest tests/ -q                               # existing suite, unchanged

# 2. smoke benchmark: all six conditions, 10 tasks, 5 seeds, stub models
python research/slm_harness/scripts/run_smoke.py --seeds 0 1 2 3 4

# 3. metrics + attribution + claim check + break-even
python research/slm_harness/scripts/compute_metrics.py \
  --runs research/slm_harness/results/smoke/runs.jsonl --finetune-cost 250

# 4. distillation dataset from successful C1/C6 trajectories
python research/slm_harness/scripts/build_distillation.py \
  --runs research/slm_harness/results/smoke/runs.jsonl \
  --tasks research/slm_harness/tasks/smoke/smoke_tasks.json

# 5. figures + numbered tables
python research/slm_harness/scripts/make_figures.py \
  --runs research/slm_harness/results/smoke/runs.jsonl

# 6. trainer dry-run (validates data/config; no GPU, no downloads)
python research/slm_harness/training/train_lora.py \
  --base-model Qwen/Qwen3-4B-Instruct-2507 \
  --train research/slm_harness/results/distillation/train.jsonl --out /tmp/ck --dry-run
```

> **Stub-run numbers are synthetic pipeline-validation output, not research
> results.** Stub clients use a seeded policy with per-profile `stub_skill`
> placeholders and chars/4 token estimates; figures and tables are watermarked
> accordingly. No research claims can be made until real-model runs exist.

## How the custom harness works (requirements → code)

1. **Constrained action space** — `CustomHarnessState.valid_actions()` exposes only
   currently-valid actions (e.g. ANSWER requires a prior READ); `verifier.validate_action`
   rejects out-of-set attempts.
2. **Minimal structured observation** — `state.build_observation()` renders compact JSON
   (goal, valid actions, recent searches/reads, last result, feedback) under a char budget.
3. **Externalized state** — goal, known files, read spans, search history, failed
   attempts, verifier feedback, budgets all live in `CustomHarnessState`, not in the model
   context. Each step is a fresh single-message call.
4. **Cheap deterministic verification** — pre-execution action validation + post-execution
   result checks + terminal answer scoring, all rule-based (`verifier.py`).
5. **Retry localization** — `CustomSearchHarness._one_step_with_retries` re-prompts only
   the failed step with feedback (`max_retries_per_step`); the task never restarts.
6. **Distillation-ready logging** — every step logs the exact observation and parsed
   action (`schemas/runlog.py`), so `distill.py` converts verified trajectories to SFT
   examples without re-running.

Final answers in ALL conditions are scored by the same deterministic scorer
(`verifier.score_final_answer`); the generic harness's free-text answer is parsed with
`extract_paths_from_text` first.

## Adding tasks

Append to a suite JSON (see `tasks/smoke/smoke_tasks.json`):

```json
{
  "task_id": "nav-011",
  "workspace": "fixture_repo",
  "prompt": "Find where `retry_with_backoff` is implemented.",
  "expected_files": ["app/util/retry.py"],
  "verification": {"method": "expected_files", "require_all_files": true},
  "split": "train"
}
```

`workspace` is resolved relative to the suite file. Verification methods:
`expected_files`, `answer_regex`, `both`. For new benchmarks (e.g. SWE-bench Verified
repos), write an adapter that materializes each instance's repo checkout as a workspace
directory and emits this same task schema — runner/harness/metrics are unchanged.

## Running real C1–C6 experiments

1. **C1/C6 (frontier model):** set the `large_general` profile to
   `provider="anthropic"`, export `ANTHROPIC_API_KEY`, set real dated pricing
   (`is_placeholder=False`).
2. **C2/C3 (small general):** serve Qwen3-4B with vLLM, set `provider="openai_compat"`
   and `base_url`. Self-host pricing = GPU-hour rate ÷ measured throughput.
3. **Distill:** run C1 (and C6) on the `train` split, then
   `scripts/build_distillation.py`.
4. **Train:** `training/README.md` — LoRA rank 32–64 on Qwen3-4B / Qwen3-1.7B
   (secondary: Llama-3.2-3B). Record actual cost into `finetune_cost_usd`.
5. **C4/C5 (fine-tuned):** serve the LoRA adapter via vLLM, point the
   `small_finetuned` profile at it.
6. **Evaluate** on `val`/`test` splits only: `run_experiment(config, splits=["test"])`.
   Decoding is temperature-0 and the nominal seed is not threaded into sampling, so
   extra seeds replay the same trajectory — spend budget on more *tasks/repos*, not
   seeds, and let `analysis.py` deduplicate any replays.
7. **Report:** `compute_metrics.py` (attribution + claim + break-even),
   `make_figures.py --real`, and `compute_significance.py` (dedup, Wilson CIs,
   exact McNemar, paired bootstrap over tasks).

## Implemented vs stubbed / not implemented

Implemented and tested: schemas; C1–C6 registry; generic + custom harnesses; deterministic
verifier; JSONL logging; runner; all metrics incl. attribution, super-additivity, claim
check, break-even; distillation builder (C6 verbatim + C1 replay); figure/table generation;
offline smoke benchmark; LoRA training (executed for real on the A40 run); the realbench
generator over real repos; the turnkey pod pipeline (`infra/run_all.sh`); statistical
analysis (`analysis.py`: dedup, Wilson CIs, exact McNemar, paired task bootstrap).

Completed for real (2026-06-10 A40 run): C1–C6 evaluation on the realbench test split;
teacher distillation; 3B LoRA training + merge; metrics, figures, significance, and the
filled white paper. Raw run logs are committed under `results/real/`.

Stubbed or placeholder (clearly labeled in code): stub-client skill levels and chars/4
token estimates (offline tests only); the data-efficiency figure (needs checkpoints
trained on nested data subsets).

Not implemented yet (next steps): the 1.5B size ablation (`RUN_ABLATION=1`, pending);
additional held-out test repos for external validity; SWE-bench Verified /
Terminal-Bench task adapters; a second subagent role (test execution & result parsing)
for the generalization phase.
