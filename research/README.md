# The SLM-Subagent Experiment, Explained in Plain English

This folder contains a research scaffold built on top of the OpenHarness/ohmo codebase.
This page explains what it does and why, using analogies. The technical version lives in
[`slm_harness/README.md`](slm_harness/README.md), and the map of how it connects to the
existing code lives in [`slm_harness/REPO_MAP.md`](slm_harness/REPO_MAP.md).

---

## The problem: your cheap intern is secretly expensive

Imagine a law firm. The senior partner (a frontier model like Claude) doesn't do every
task personally — that would be absurdly expensive. Instead, the firm hires generalist
interns (cheap general-purpose models) to do the legwork: find documents, look things up,
run errands.

Here's the catch. Each intern shows up to every errand carrying the firm's entire
employee handbook, a master key ring with forty keys they'll never use, and no notepad —
they keep everything in their head. They're paid a low hourly rate, but:

- they re-read the whole handbook on every trip (huge prompts, every call),
- they fumble with keys for doors they don't need (large tool lists to choose from),
- they forget things and have to start errands over (free-form context, no externalized
  state),
- when they bring back the wrong document, the senior partner has to notice, explain the
  mistake, and send them again (retries and orchestrator corrections).

So the *hourly rate* is low, but the *cost per completed errand* is high. That's the
trap this research targets: in today's multi-agent systems, "cheap" general worker
models are not actually cheap once you divide total spend by tasks that actually
succeeded.

## The idea: a specialist with a perfectly-designed workstation

Instead of a generalist intern, train one person to do exactly ONE job — say, fetching
files from the records room — and redesign the workstation around them:

- **A menu instead of an open question.** They never decide "what should I do next?"
  from scratch. The workstation shows only the moves that are legal right now: SEARCH,
  READ, ANSWER, or ESCALATE ("get the partner"). You can't press a button that doesn't
  apply.
- **A clipboard instead of memory.** The goal, what's been searched, which files were
  found, what failed — all of it lives on a clipboard the *workstation* maintains. The
  specialist reads one small index card per step and never has to remember anything.
- **A checker at the door.** After every step, a deterministic checker (no AI, just
  rules) verifies the work: does that file actually exist? Do the cited line numbers
  actually exist? Did you cite a file you never actually opened?
- **Fix the step, not the errand.** If step 3 goes wrong, the checker hands back step 3
  with a note saying exactly what was wrong. The specialist redoes step 3 — not the
  whole errand.
- **Everything is filmed for training.** Every index card shown and every move made is
  logged in exactly the format you'd use to train the next specialist.

That redesigned workstation is the **custom harness**. The specialist is a **small
language model (SLM)** — think Qwen3-4B, ~100x smaller than a frontier model — and
"training the specialist" is **fine-tuning** the SLM on recordings of the senior partner
doing the same job at this exact workstation (**distillation**).

**The claim:** specialist + workstation cuts cost-per-successful-task by at least 50%
versus the senior partner doing it themselves, while losing at most 2 percentage points
of success rate.

## The experiment: six hiring policies, tested fairly

A skeptical reviewer will ask: "if it works, was it the specialist (model fine-tuning) or
the workstation (harness)? You changed two things at once." This is currently the #1
critique in the agents literature — papers crediting the model for gains that came from
the harness.

So we test every combination, like a proper controlled trial. Two workstations × three
workers:

| | Generic workstation (handbook + key ring) | Custom workstation (menu + clipboard + checker) |
|---|---|---|
| **Senior partner** (large model) | **C1** — how firms work today (baseline) | **C6** — best case: expert + great station |
| **Generalist intern** (small model) | **C2** — the naive "just hire cheaper" move | **C3** — does the workstation alone help? |
| **Trained specialist** (fine-tuned small model) | **C4** — does training alone help? | **C5** — the proposed system |

This 2×3 grid lets us compute, separately:

- **The workstation effect:** C3 − C2 (same intern, better station)
- **The training effect:** C4 − C2 (same station, trained intern)
- **The combination effect:** if workstation and training were independent, C5 should
  score about C3 + C4 − C2. If C5 beats that prediction, the two interventions are
  **super-additive** — the specialist is *extra* effective precisely because the
  workstation speaks the exact language they were trained in. That interaction term is
  the scientific heart of the paper.

And the prediction everyone underestimates: **C2 should lose to C1 on
cost-per-successful-task even though its tokens cost ~30x less** — because the generalist
intern fails so often that retries eat the savings. That's what makes the full system
non-obvious: the cheap option everyone reaches for first doesn't actually work.

## The job we test first

Phase 1 uses **file/code search and navigation** ("find which file defines
`RateLimiter`") as the subagent job. It's the records-room errand of coding agents:
extremely frequent, requires little reasoning, naturally scoped, and gradeable by
deterministic rules (the file either is or isn't the right one). Later phases extend to
established benchmarks (SWE-bench Verified-style tasks) and a second job to show the
recipe generalizes.

## How the money is counted

We don't compare hourly rates; we compare **cost per successful errand**:

> total dollars spent on a condition ÷ number of tasks that actually succeeded

Failures still cost money — they just produce nothing, which is exactly why "cheap but
flaky" loses. Training the specialist costs a one-time fee, so there's also a
**break-even analysis**: at N tasks per month, how many weeks until the fine-tuning
investment pays for itself?

## What's in this folder (the assembly line)

```
slm_harness/
  schemas/        the paperwork: typed definitions of models, conditions, tasks, logs
  conditions.py   the six hiring policies, C1–C6
  harness/        the two workstations (generic.py, custom.py) + the clipboard (state.py)
  verifier.py     the checker at the door (pure rules, no AI)
  runner.py       the experiment conductor: runs every (policy × task × seed), logs JSONL
  metrics.py      the accountant: success rates, cost-per-success, the 2×3 decomposition
  distill.py      the film-to-textbook converter: partner recordings → SLM training data
  training/       how to train the specialist (LoRA on Qwen3-4B); not auto-executed
  figures.py      paper-ready charts and numbered tables
  tasks/smoke/    a tiny fake codebase + 10 navigation tasks for offline pipeline tests
  tests/          44 tests covering all of the above
```

It's built as an **additive layer**: the existing OpenHarness machinery (tools, token
accounting, message formats, provider clients) is reused as-is — both workstations even
search and read files through the *same* underlying tools, so the only experimental
variable is what the worker sees.

## What the numbers in `results/` are (and are not)

The committed smoke results were produced with **stub workers** — scripted stand-ins with
dialed-in skill levels — to prove the assembly line works end to end: tasks run, the
checker checks, retries localize, logs distill into valid training files, charts render.
Every figure and table from these runs is watermarked **SYNTHETIC**. They demonstrate the
pipeline, not the hypothesis. No research claims exist until real models run; the README
in `slm_harness/` has the exact recipe for that.

## Try it (offline, no API keys, ~1 minute)

```bash
pip install -e ".[dev]"
pytest research/slm_harness/tests/ -q                       # 44 tests
python research/slm_harness/scripts/run_smoke.py            # all 6 conditions, stub workers
python research/slm_harness/scripts/compute_metrics.py \
  --runs research/slm_harness/results/smoke/runs.jsonl --finetune-cost 250
python research/slm_harness/scripts/make_figures.py \
  --runs research/slm_harness/results/smoke/runs.jsonl
```
