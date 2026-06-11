# Specialized Small-Model Subagents: Co-Designing a Fine-Tuned SLM and a Task-Specific Harness for Cost-Efficient Multi-Agent Systems

**Status:** FINAL (real self-hosted run, 2026-06-10)
**Authors:** Ishaan Ranjan (Luxen LLC / AgentTree) · *affiliations TBD*
**Artifact:** `github.com/IshaanAyaan/slm-agents` (branch `main`), `research/slm_harness/`

---

## Abstract

Production multi-agent systems pair a frontier-model orchestrator with cheaper
general-purpose models as worker subagents. These workers remain expensive in practice:
they carry large prompts and tool lists, accumulate unnecessary context, and fail often,
triggering retries and orchestrator corrections that erase their per-token savings. We
test whether a **small language model fine-tuned for one subagent role, paired with a
co-designed task-specific harness**, beats a cheap general-purpose subagent on
**cost-per-successful-task** while preserving success rate. Using a controlled 2×3
factorial design over {model ∈ large-general, small-general, small-fine-tuned} ×
{harness ∈ generic, custom}, we isolate the harness effect, the fine-tuning effect, and
their interaction. On a deterministic file/code-navigation benchmark with automatically
verified ground truth, our proposed system (C5) achieves a **74.9%**
reduction in cost-per-successful-task versus the standard large-model baseline (C1) while
**increasing** success rate by **25.0 pp** (0.725 → 0.975) — exceeding the pre-registered
target of a ≥50% cost reduction at ≤2 pp success drop. The fine-tuning×harness interaction
is strongly super-additive on success rate (+0.650 over the additive prediction),
confirming that the SLM and harness must be co-designed rather than combined post hoc. The
entire study runs on self-hosted open models with **no paid API tokens**; cost is grounded
in measured GPU throughput.

---

## 1. Introduction

The dominant pattern in agentic systems is hierarchical: a strong orchestrator decomposes
a task and delegates sub-tasks to worker agents. To control cost, practitioners swap the
expensive frontier worker for a cheaper general-purpose model. This paper argues, and
shows, that the naive swap is a trap — and that a better option exists.

A cheap general model used zero-shot as a subagent still receives the orchestrator's broad
system prompt, a large tool list, and free-form accumulated context; it reasons in ways
that are unnecessary for a narrow role; and it fails frequently. Each failure is not free:
it costs the worker's own tokens plus the orchestrator's tokens to detect and correct the
error and re-delegate. Measured by **cost-per-successful-task** rather than cost-per-token,
the cheap general worker can be *worse* than the model it replaced.

We propose replacing the general worker with a **specialist**: a small model fine-tuned for
exactly one role, operating inside a **custom harness** that (1) constrains the action
space to the moves valid at the current step, (2) compresses raw environment state into a
minimal structured observation, (3) externalizes all durable state (goal, history,
discovered artifacts, failed attempts) so the model carries almost no free-form context,
(4) runs cheap deterministic verification after every step, and (5) re-prompts only the
failed step rather than restarting the task. The harness and the model are **co-designed**:
the model is fine-tuned, by distillation from frontier-model trajectories, to speak exactly
the harness's input/output schema.

**Contributions.**
1. A method for co-designing a fine-tuned SLM and a task-specific harness for a subagent
   role, with a distillation pipeline that solves the role-specific training-data problem
   by reusing successful orchestrator trajectories.
2. A controlled 2×3 factorial attribution that separates harness, fine-tuning, and
   interaction effects — addressing the field's harness-disclosure confound directly.
3. Empirical cost-per-successful-task results showing where specialized subagents win and
   where the naive cheap option fails, obtained with **zero API spend** via self-hosted
   models and a GPU-throughput-grounded cost model.

---

## 2. Related work and motivation

Three findings motivate this design. First, **harness quality can dominate model choice**:
holding the model fixed and changing only the scaffold can move benchmark scores by
double-digit points. Second, **fine-tuned small models match or beat much larger general
models on narrow tasks**, often at a fraction of the cost. Third, **reasoning belongs at
the orchestrator, not the subagent**: enabling subagent reasoning yields limited or
negative benefit, which is precisely the regime where tightly-harnessed SLMs win.

The gap: existing "cheap subagent" systems substitute cheaper *general* models with
*generic* harnesses and rarely disclose how much of any gain came from the harness.
Reviewers increasingly flag this confound. Our 2×3 design is built specifically to resolve
it.

---

## 3. Method

### 3.1 System and the fixed orchestrator

A single fixed orchestrator delegates one navigation task to the subagent under test and
accepts its result (answer, escalation, or failure). Holding the orchestrator fixed and
deterministic removes orchestrator variance from the comparison; the subagent layer is the
only variable.

### 3.2 The 2×3 factorial design

|  | generic harness | custom harness |
|---|---|---|
| large general model | **C1** baseline | **C6** upper bound |
| small general model | **C2** naive cost-cut | **C3** harness-only |
| small fine-tuned model | **C4** fine-tuning-only | **C5** proposed |

From the small-model 2×2 sub-grid we compute, on any metric:
harness effect = C3−C2, fine-tuning effect = C4−C2, additive prediction for C5 = C3+C4−C2,
and **interaction = C5 − (C3+C4−C2)**. A positive interaction on success (negative on cost)
means the two interventions are super-additive — the hallmark of genuine co-design.

### 3.3 The custom harness

The Phase-1 role is file/code search and navigation. The custom harness exposes four
actions — `SEARCH(pattern, file_glob, root)`, `READ(path, offset, limit)`,
`ANSWER(files, evidence, confidence)`, `ESCALATE(reason)` — and maintains all state
externally. Each step the model receives a compact JSON observation (goal, valid actions,
recent searches/reads, last result, and any verifier feedback) and must emit exactly one
JSON action. A deterministic verifier checks each action *before* execution (valid action
for this step? real in-workspace path? compilable regex? answer cites only discovered
files, with in-range evidence spans?) and re-prompts only the failed step. `SEARCH`/`READ`
execute through the same underlying file tools as the generic harness, so the environment
is held constant and only the model-facing interface varies.

### 3.4 Distillation pipeline

We run the large model inside both harnesses on the training split and keep only
trajectories that pass the verifier and the final task scorer. Custom-harness (C6) steps
are taken verbatim as (observation → action) pairs; generic-harness (C1) trajectories are
*replayed* through the custom-harness state machine so each tool call is re-expressed as a
schema-valid action with the exact observation the SLM will see at inference. The result is
chat-format SFT JSONL, split train/val/test, used to LoRA-fine-tune the SLM (rank 64).

---

## 4. Experimental setup

### 4.1 Zero-API, self-hosted execution

All models are served locally with vLLM 0.7.3 on a single **NVIDIA A40 (48 GB, $0.44/hr)**.
The large/teacher model is **Qwen2.5-14B-Instruct**; the student is **Qwen2.5-3B-Instruct**,
LoRA-fine-tuned (rank 64, 3 epochs) on 234 distilled training examples (24 val) and served
merged. No paid API is used anywhere. One model is resident at a time; conditions run
sequentially against the same card, so throughput-derived costs are directly comparable.

### 4.2 Cost model

Cost is grounded in measured hardware throughput, not an external price list. For each
served model we benchmark realized prefill/decode tokens-per-second and convert
`GPU_count × $/GPU-hour` into per-token USD. The 14B teacher is therefore correctly
costlier per token than the 3B student. Fine-tuning cost is measured training
wall-time × GPUs × rate ($0.05 for this run), used in the break-even analysis. Because
every condition shares the same GPU and rate, the headline *ratios* are independent of
the chosen $/GPU-hour.

### 4.3 Benchmark

Tasks are generated deterministically from real Python repositories: we index top-level
class/function/constant definitions and keep symbols defined in exactly one file, yielding
"which file defines `X`?" tasks with unambiguous, machine-checkable ground truth and **no
human labeling**. Train-split repos (OpenHarness `src/`, `requests`; 120 tasks) feed
distillation; a held-out test-split repo (`ohmo`; 40 tasks) measures generalization to
code the SLM never trained on. Every condition runs each test task with 2 seeds
(80 attempts per condition).

---

## 5. Results

> Numbers below are auto-filled from `results/real/metrics_a.json` (run of 2026-06-10).

### 5.1 Main results (headline, Qwen2.5-3B specialist)

| Condition | Success rate | Cost / successful task (USD) | Cost / attempt (USD) |
|---|---|---|---|
| C1 large+generic (baseline) | 0.725 | 0.001818 | 0.001318 |
| C2 small+generic (naive cut) | 0.463 | 0.003820 | 0.001767 |
| C3 small+custom (harness-only) | 0.263 | 0.002132 | 0.000560 |
| C4 fine-tuned+generic (FT-only) | 0.525 | 0.002291 | 0.001203 |
| **C5 fine-tuned+custom (proposed)** | **0.975** | **0.000456** | **0.000445** |
| C6 large+custom (design-stage upper bound; see §5.4) | 0.550 | 0.003762 | 0.002069 |

![Cost per successful task](research/slm_harness/results/real/figures_a/fig2_cost_per_success.png)
![Success by condition](research/slm_harness/results/real/figures_a/fig1_success_by_condition.png)

### 5.2 Primary claim (C5 vs C1)

Cost-per-successful-task reduction: **74.9%** (target ≥ 50%).
Success-rate change: **+25.0 pp** (0.725 → 0.975; target allowed up to a 2 pp drop).
**Claim holds — and the specialist is more reliable than the baseline, not merely cheaper.**

### 5.3 The naive cheap option fails (C2 vs C1)

A key secondary result: despite far cheaper tokens, C2's cost-per-successful-task
(**$0.003820**) is **2.1× worse** than C1's (**$0.001818**) — retries and corrections eat
the per-token savings. C2 also consumes 3.6× the input tokens per attempt (11,063 vs
3,074 mean), exactly the failure-amplification the introduction predicts. This is what
makes the full system non-obvious: swapping in a cheap general model is a net loss.

### 5.4 Attribution decomposition

![Attribution](research/slm_harness/results/real/figures_a/fig3_attribution_success.png)

On **success rate**: harness effect (C3−C2) = **−0.200**;
fine-tuning effect (C4−C2) = **+0.062**;
additive prediction for C5 = 0.325;
observed C5 = 0.975, i.e. interaction = **+0.650**; **super-additive**.

On **cost-per-successful-task**: harness effect = −$0.001688;
fine-tuning effect = −$0.001530;
interaction = −$0.000147; **super-additive** (more negative cost than additive).

The decomposition is the scientific core of the result. The custom harness *hurts* the
untrained small model (C3 < C2: the base 3B fails the strict JSON action schema, averaging
2.5 invalid actions per attempt) and even hurts the large model (C6 = 0.550 < C1 = 0.725,
with 1.9 invalid actions per attempt — the "upper bound" condition lands *below* the
generic baseline). Fine-tuning alone barely helps (C4 ≈ C2 + 6 pp). Only the combination —
a model distilled specifically to speak the harness's schema — reaches 0.975 with a 0.025
invalid-action rate. Neither intervention works without the other; this is co-design, not
stacking.

### 5.5 Break-even on the one-time fine-tuning cost

![Break-even](research/slm_harness/results/real/figures_a/fig5_break_even.png)

Fine-tuning cost = $0.05; savings per successful task = $0.001361;
**break-even at 37 successful tasks**. At realistic subagent volumes the one-time
fine-tuning cost is negligible.

### 5.6 Size ablation (Qwen2.5-1.5B specialist)

The 1.5B ablation was disabled in this budget run (`RUN_ABLATION=0`); it tests how far
the model can shrink before the harness can no longer carry it, and is left to a
follow-up run.

---

## 6. Discussion

The practical implication is that the cheapest reliable subagent is not a smaller general
model but a **co-designed specialist**: a tiny fine-tuned model whose harness owns state,
memory, valid actions, and verification. The super-additivity result is the scientific
core — it shows the two interventions are not independent knobs; the model becomes
extra-effective precisely because it was trained to speak the harness's language. The C5
specialist is simultaneously the **most reliable** (0.975) and the **cheapest per success**
($0.000456) of all six conditions, while the naive cheap swap (C2) is the most expensive
per success after C6 — the two ends of the design space the field currently conflates.

## 7. Limitations and threats to validity

- **One role, one task family (Phase 1).** Generalization to a second role (e.g. test
  execution and result parsing) and to SWE-bench-Verified-style tasks is future work.
- **Open 14B teacher as frontier proxy.** Cost is API-free by using Qwen2.5-14B-Instruct
  as the large/teacher condition; a frontier API model would likely raise C1's success
  rate (and its cost). An optional small real-frontier slice can anchor representativeness.
- **Deterministic navigation ground truth** is unambiguous but narrower than open-ended
  agentic tasks; it is chosen for measurement rigor in Phase 1.
- **Single test repository, 40 tasks × 2 seeds.** Error bars are reported (C5 stderr
  0.017, others ~0.05); more repos and seeds would tighten them.
- **Cost model** reflects measured throughput on one A40 and a chosen GPU rate
  ($0.44/hr); absolute dollars scale with both, though the *ratios* between conditions
  are rate-independent because all conditions share the same hardware.
- **Harness prompt iteration.** The custom-harness prompt was refined once during the
  run (stating the READ-before-ANSWER rule) after observing the teacher burning retries;
  the change applies uniformly to every custom-harness condition (C3, C5, C6).

## 8. Reproducibility

All harness specifications, the distillation pipeline, the deterministic benchmark
generator, training configs, evaluation scripts, and this paper's number-filling script are
released in `research/slm_harness/`. The full run is a single command
(`infra/run_all.sh`) on one self-hosted A40/A6000-class GPU with no API keys; this run
completed in under an hour of GPU time (~$0.50 at the quoted rate).

## 9. Conclusion

We tested whether a fine-tuned SLM with a co-designed harness can be a cheaper-yet-reliable
subagent than a general cheap model under a fixed orchestrator. Under a controlled 2×3
attribution, the proposed system delivers a 74.9% cost-per-successful-task
reduction versus the standard baseline while raising success rate by 25 pp (0.725 → 0.975),
with a strongly super-additive fine-tuning×harness interaction (+0.650 over the additive
prediction). Specialized subagents, not merely smaller ones, are the cost-efficient
building block for multi-agent systems.

---

*Appendix A — full per-condition token/reliability table and Tables 1–4 are generated in
`research/slm_harness/results/real/figures_a/tables.md`. Appendix B — per-task trajectories
are in `results/real/eval_teacher/runs.jsonl` (C1, C6), `results/real/eval_a_base/runs.jsonl`
(C2, C3), and `results/real/eval_a_ft/runs.jsonl` (C4, C5).*
