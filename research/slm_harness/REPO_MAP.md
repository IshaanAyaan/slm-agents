# REPO_MAP — How the research scaffold relates to the existing OpenHarness/ohmo codebase

This document maps the existing infrastructure repo and explains exactly which pieces the
`research/slm_harness/` scaffold reuses, which it deliberately re-implements, and what must not
be touched.

## 1. Current repo architecture

The repo is **openharness-ai** (`pyproject.toml`, v0.1.9): an open-source Python port of a
Claude-Code-style agentic CLI, plus `ohmo`, a gateway/orchestration layer.

```
src/openharness/
  api/        Provider abstraction. AnthropicApiClient (retry/streaming), openai_client,
              copilot_client, registry (provider detection), usage.UsageSnapshot
  engine/     The agent loop. query.run_query + QueryContext (tool loop, auto-compaction),
              query_engine.QueryEngine (history + memory + hooks), messages (ConversationMessage,
              TextBlock/ToolUseBlock/ToolResultBlock), cost_tracker.CostTracker, stream_events
  tools/      Tool registry + ~40 tools. base.BaseTool/ToolRegistry/ToolResult/
              ToolExecutionContext; file_read_tool, grep_tool, glob_tool, bash_tool,
              file_edit/write, web tools, task/team/agent (subagent) tools, …
  permissions/ PermissionChecker, sensitive-path denylist, modes
  hooks/      HookExecutor + event schema (pre/post tool use, prompt submit, …)
  swarm/      Subagent spawning: in-process + subprocess backends, mailbox, team lifecycle
  tasks/      Background task manager (local agent/shell tasks)
  services/   tool_outputs (output truncation/offload budgets), compact (auto-compaction),
              session storage, token_estimation, cron, …
  config/     Settings (pydantic), paths
  prompts/    System prompt assembly (system_prompt.py, context.py, claudemd.py)
  ui/, channels/, voice/, plugins/, skills/, memory/, …  (CLI/UX layers — not used here)
ohmo/         Gateway runtime/CLI on top of openharness (not used here)
tests/        Existing test suite for all of the above (pytest, testpaths=["tests"])
```

## 2. Where the research scaffold lives

Everything new is **additive** under `research/slm_harness/`:

```
research/slm_harness/
  REPO_MAP.md            this file
  README.md              research question, how to run everything
  schemas/               typed pydantic models: config, tasks, actions, run logs
  conditions.py          C1–C6 condition registry (2x3 factorial)
  model_clients/         model-call abstraction: protocol, Anthropic adapter,
                         OpenAI-compatible adapter (vLLM/SLM), deterministic stub clients
  harness/               generic harness adapter + custom search/navigation harness + state
  verifier.py            deterministic step verifier + final task scorer
  runner.py              experiment runner, JSONL logging, cost accounting
  metrics.py             metrics + 2x3 attribution decomposition + super-additivity + break-even
  distill.py             distillation dataset builder (C1/C6 trajectories → SFT JSONL)
  figures.py             figure + numbered-table generation
  tasks/smoke/           local smoke benchmark: synthetic fixture repo + task JSON
  training/              LoRA training interface + README (not executed without deps/keys)
  scripts/               CLI entry points (run_smoke, compute_metrics, make_figures, build_distillation)
  tests/                 research-scaffold tests (run separately from the main suite)
```

## 3. Existing modules that are reused (imported directly)

| Existing module | What we reuse | Where |
|---|---|---|
| `openharness.tools.base` | `BaseTool`, `ToolRegistry`, `ToolExecutionContext`, `ToolResult` | generic harness tool loop; custom harness executes actions through the same tool implementations |
| `openharness.tools.file_read_tool` | `FileReadTool` (line-numbered reads, offset/limit) | both harnesses (custom `READ` action delegates to it) |
| `openharness.tools.grep_tool` | `GrepTool` (rg + pure-python fallback) | both harnesses (custom `SEARCH` action delegates to it) |
| `openharness.tools.glob_tool` | `GlobTool` | generic harness tool list; custom `SEARCH` file_glob |
| `openharness.api.usage` | `UsageSnapshot` token accounting model | all model clients, cost accounting |
| `openharness.engine.cost_tracker` | `CostTracker` accumulation | runner per-attempt accounting |
| `openharness.engine.messages` | `ConversationMessage`, `TextBlock`, `ToolUseBlock`, `ToolResultBlock` | generic harness conversation state; Anthropic adapter |
| `openharness.api.client` | `ApiMessageRequest`, `ApiMessageCompleteEvent`, `SupportsStreamingMessages` protocol, `AnthropicApiClient` | real-model Anthropic adapter |
| `openharness.services.tool_outputs` | inline/preview char budgets | generic harness tool-output truncation (mirrors production offloading behavior) |

## 4. Existing modules deliberately *not* imported (and why)

- `engine.query.run_query` / `QueryEngine`: this is the production interactive loop. It is
  coupled to permission prompts, hooks, auto-compaction, session memory, autodream, and
  coordinator context. For a *measurement* harness we need a loop that is deterministic,
  fully instrumented (per-call token/latency capture), and seedable. The generic harness
  (`harness/generic.py`) therefore implements a minimal tool loop with the **same semantics**
  (assistant turn → execute tool_use blocks → tool_result user message → repeat) using the same
  message/tool abstractions, so its behavior is representative of the production loop while
  remaining reproducible. This is documented as a methodological choice, not a fork.
- `swarm/` subagent spawning: process-level isolation is orthogonal to the hypothesis; the
  runner plays the role of the fixed orchestrator deterministically (see README §design).
- `permissions/`: eval workspaces are read-only fixtures; both harnesses only run read-only
  tools, so permission UX is out of scope and bypassed uniformly across conditions.
- `ohmo/`, `ui/`, `channels/`: interactive/gateway layers, irrelevant to the experiment.

## 5. What must not be touched

- Nothing under `src/openharness/`, `ohmo/`, `tests/`, `scripts/`, `frontend/`,
  `autopilot-dashboard/` is modified. The research scaffold only imports from
  `src/openharness/`.
- `pyproject.toml` is unmodified; research code is run from the repo root with
  `PYTHONPATH=src` (the scripts and tests set this up automatically via `research/slm_harness/_bootstrap.py`).
- Existing tests still run as before: `pytest tests/`. Research tests run with
  `pytest research/slm_harness/tests/`.

## 6. How the research code connects to the existing harness

- **Generic harness condition (C1/C2/C4)** = ohmo-style subagent: broad system prompt
  (assembled in the spirit of `openharness.prompts`), the full general-purpose read-only tool
  list from the ohmo registry (`read_file`, `grep`, `glob`, plus distractor tool schemas to
  reproduce realistic prompt bloat), free-form context accumulation in `ConversationMessage`
  history, production-style tool-output truncation budgets.
- **Custom harness condition (C3/C5/C6)** = the co-designed narrow harness: 4 actions
  (`SEARCH`/`READ`/`ANSWER`/`ESCALATE`), minimal JSON observations, externalized state,
  per-step deterministic verification, failed-step-only retry. Its `SEARCH`/`READ` executors
  call the *same* `GrepTool`/`FileReadTool` implementations as the generic harness, so the
  environment is held constant and only the model-facing interface varies.
- **Model access** goes through `model_clients/`: the Anthropic adapter wraps
  `openharness.api.client.AnthropicApiClient` (the production provider abstraction); the
  OpenAI-compatible adapter targets vLLM/llama.cpp-style endpoints for SLMs; stub clients allow
  the full pipeline (runner → harness → verifier → logs → metrics → figures) to execute with no
  network access or credentials.
