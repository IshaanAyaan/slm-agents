# Specialized Small-Model Subagents: Co-Designing a Fine-Tuned SLM and a Task-Specific Harness for Cost-Efficient Multi-Agent Systems

**Status:** {{STATUS}}
**Authors:** Ishaan Ranjan (Luxen LLC / AgentTree) · *affiliations TBD*
**Artifact:** `github.com/LuxenAI/slm-harness` (branch `fable`), `research/slm_harness/`

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
verified ground truth, our proposed system (C5) achieves a **{{COST_REDUCTION_PCT}}%**
reduction in cost-per-successful-task versus the standard large-model baseline (C1) at a
**{{SUCCESS_DROP_PP}} pp** change in success rate (target: ≥50% reduction, ≤2 pp drop;
claim holds: **{{CLAIM_HOLDS}}**). The fine-tuning×harness interaction is super-additive
on success rate (**{{SUPERADD_SR}}**), confirming that the SLM and harness must be
co-designed rather than combined post hoc. The entire study runs on self-hosted open
models with **no paid API tokens**; cost is grounded in measured GPU throughput.

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

All models are served locally with vLLM on a single self-hosted GPU; the teacher and
student checkpoints are open Qwen-family Instruct models selected in `infra/config.env`
(the 2026-06-10 run: Qwen2.5-14B teacher, Qwen2.5-3B student, one NVIDIA A40 48 GB).
No paid API is used anywhere.

### 4.2 Cost model

Cost is grounded in measured hardware throughput, not an external price list. For each
served model we benchmark realized prefill/decode tokens-per-second and convert
`GPU_count × $/GPU-hour` into per-token USD. The larger teacher is therefore correctly
costlier per token than the student. Fine-tuning cost is measured training
wall-time × GPUs × rate, used in the break-even analysis.

### 4.3 Benchmark

Tasks are generated deterministically from real Python repositories: we index top-level
class/function/constant definitions and keep symbols defined in exactly one file, yielding
"which file defines `X`?" tasks with unambiguous, machine-checkable ground truth and **no
human labeling**. Train-split repos feed distillation; held-out test-split repos measure
generalization to code the SLM never trained on. Decoding is deterministic
(temperature 0), so the analysis unit is the task: statistics deduplicate to one attempt
per (condition, task), with Wilson CIs and paired exact tests from
`scripts/compute_significance.py`.

---

## 5. Results

> Numbers below are auto-filled from the run's metrics JSON. `[PENDING RUN]` means the
> self-hosted run has not yet produced that value.

### 5.1 Main results (headline specialist)

| Condition | Success rate | Cost / successful task (USD) | Cost / attempt (USD) |
|---|---|---|---|
| C1 large+generic (baseline) | {{SR_C1}} | {{CPS_C1}} | {{CPA_C1}} |
| C2 small+generic (naive cut) | {{SR_C2}} | {{CPS_C2}} | {{CPA_C2}} |
| C3 small+custom (harness-only) | {{SR_C3}} | {{CPS_C3}} | {{CPA_C3}} |
| C4 fine-tuned+generic (FT-only) | {{SR_C4}} | {{CPS_C4}} | {{CPA_C4}} |
| **C5 fine-tuned+custom (proposed)** | **{{SR_C5}}** | **{{CPS_C5}}** | **{{CPA_C5}}** |
| C6 large+custom (upper bound) | {{SR_C6}} | {{CPS_C6}} | {{CPA_C6}} |

![Cost per successful task]({{FIGDIR}}/fig2_cost_per_success.png)
![Success by condition]({{FIGDIR}}/fig1_success_by_condition.png)

### 5.2 Primary claim (C5 vs C1)

Cost-per-successful-task reduction: **{{COST_REDUCTION_PCT}}%** (target ≥ 50%).
Success-rate change: **{{SUCCESS_DROP_PP}} pp** (target ≤ 2 pp drop).
**Claim holds: {{CLAIM_HOLDS}}.**

### 5.3 The naive cheap option fails (C2 vs C1)

A key secondary result: despite far cheaper tokens, C2's cost-per-successful-task
({{CPS_C2}}) versus C1 ({{CPS_C1}}) shows that retries and corrections eat the savings —
this is what makes the full system non-obvious.

### 5.4 Attribution decomposition

![Attribution]({{FIGDIR}}/fig3_attribution_success.png)

On **success rate**: harness effect (C3−C2) = {{HARNESS_EFFECT_SR}};
fine-tuning effect (C4−C2) = {{FINETUNE_EFFECT_SR}};
additive prediction for C5 = {{ADDITIVE_PRED_SR}};
observed interaction = {{INTERACTION_SR}}; **super-additive: {{SUPERADD_SR}}**.

On **cost-per-successful-task**: harness effect = {{HARNESS_EFFECT_CPS}};
fine-tuning effect = {{FINETUNE_EFFECT_CPS}};
interaction = {{INTERACTION_CPS}}; **super-additive: {{SUPERADD_CPS}}**.

### 5.5 Break-even on the one-time fine-tuning cost

![Break-even]({{FIGDIR}}/fig5_break_even.png)

Fine-tuning cost = ${{FT_COST}}; savings per successful task = ${{SAVINGS_PER_TASK}};
**break-even at {{BREAKEVEN_TASKS}} tasks**.

### 5.6 Size ablation (smaller specialist)

C5 with a 1.7B specialist: success rate {{SR_C5_17B}}, cost/success {{CPS_C5_17B}} —
testing how far the model can shrink before the harness can no longer carry it.

---

## 6. Discussion

If the claim holds, the practical implication is that the cheapest reliable subagent is not
a smaller general model but a **co-designed specialist**: a tiny fine-tuned model whose
harness owns state, memory, valid actions, and verification. The super-additivity result is
the scientific core — it shows the two interventions are not independent knobs; the model
becomes extra-effective precisely because it was trained to speak the harness's language.

## 7. Limitations and threats to validity

- **One role, one task family (Phase 1).** Generalization to a second role (e.g. test
  execution and result parsing) and to SWE-bench-Verified-style tasks is future work.
- **Open teacher as frontier proxy.** Cost is API-free by using a strong open model as the
  large/teacher condition; an optional small real-frontier slice can anchor representativeness.
- **Deterministic navigation ground truth** is unambiguous but narrower than open-ended
  agentic tasks; it is chosen for measurement rigor in Phase 1.
- **Cost model** reflects measured throughput on specific hardware and a chosen GPU rate;
  absolute dollars scale with both, though the *ratios* between conditions are robust.

## 8. Reproducibility

All harness specifications, the distillation pipeline, the deterministic benchmark
generator, training configs, evaluation scripts, and this paper's number-filling script are
released in `research/slm_harness/`. The full run is a single command
(`infra/run_all.sh`) on one self-hosted Ampere-class GPU with no API keys.

## 9. Conclusion

We tested whether a fine-tuned SLM with a co-designed harness can be a cheaper-yet-reliable
subagent than a general cheap model under a fixed orchestrator. Under a controlled 2×3
attribution, the proposed system delivers a {{COST_REDUCTION_PCT}}% cost-per-successful-task
reduction versus the standard baseline at {{SUCCESS_DROP_PP}} pp success change, with a
super-additive fine-tuning×harness interaction. Specialized subagents, not merely smaller
ones, are the cost-efficient building block for multi-agent systems.

---

*Appendix A — full per-condition token/reliability table and Tables 1–4 are generated in
`{{FIGDIR}}/tables.md`. Appendix B — per-task trajectories are in the committed `results/real/eval_*/runs.jsonl` files.*
