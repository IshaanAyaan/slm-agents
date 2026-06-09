"""Offline stub model clients.

These exist so the ENTIRE pipeline (runner -> harness -> tools -> verifier -> JSONL
logs -> metrics -> figures) can execute with no network access, credentials, or GPUs.

``OracleStubClient`` implements a deterministic, seeded policy for the navigation
role. It does NOT see task answer keys: it extracts keywords from the goal, runs real
searches through the harness, and answers from real tool results. A ``skill``
parameter injects seeded mistakes (invalid JSON, disallowed actions, distractor tool
calls) so retry/verification/cost paths are exercised.

IMPORTANT: numbers produced with stub clients are synthetic pipeline-validation
numbers, NOT research results. Token counts are chars/4 estimates.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field

from research.slm_harness import _bootstrap  # noqa: F401

from openharness.api.usage import UsageSnapshot
from openharness.engine.messages import (
    ConversationMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)

from research.slm_harness.model_clients.base import (
    ModelCallRequest,
    ModelCallResponse,
    estimate_tokens,
)

_KEYWORD_RES = [
    re.compile(r"`([^`]+)`"),
    re.compile(r"\"([^\"]+)\""),
    re.compile(r"'([^']+)'"),
]
_GREP_HIT_RE = re.compile(r"^([\w./\-]+):(\d+):", re.MULTILINE)


def extract_keywords(goal: str) -> list[str]:
    """Pull search keywords from a task prompt: quoted terms first, then long tokens."""
    keywords: list[str] = []
    for rx in _KEYWORD_RES:
        for m in rx.findall(goal):
            term = m.strip()
            if term and term not in keywords:
                keywords.append(term)
    tokens = sorted(
        {t.strip(".,;:!?()") for t in goal.split() if len(t.strip(".,;:!?()")) > 5},
        key=len,
        reverse=True,
    )
    for t in tokens:
        if t not in keywords and not t.startswith("("):
            keywords.append(t)
    return keywords or [goal.split()[0] if goal.split() else "TODO"]


def _request_input_tokens(request: ModelCallRequest) -> int:
    """chars/4 estimate over system prompt + messages + tool schemas."""
    total = len(request.system_prompt)
    for msg in request.messages:
        for block in msg.content:
            if isinstance(block, TextBlock):
                total += len(block.text)
            elif isinstance(block, ToolResultBlock):
                total += len(block.content)
            elif isinstance(block, ToolUseBlock):
                total += len(json.dumps(block.input))
    if request.tools:
        total += len(json.dumps(request.tools))
    return max(1, total // 4)


@dataclass
class ScriptedClient:
    """Returns canned outputs in order. For tests."""

    outputs: list[str | ConversationMessage]
    calls: list[ModelCallRequest] = field(default_factory=list)

    async def complete(self, request: ModelCallRequest) -> ModelCallResponse:
        """Pop the next scripted output."""
        self.calls.append(request)
        if not self.outputs:
            raise RuntimeError("ScriptedClient exhausted")
        nxt = self.outputs.pop(0)
        message = (
            nxt
            if isinstance(nxt, ConversationMessage)
            else ConversationMessage(role="assistant", content=[TextBlock(text=nxt)])
        )
        return ModelCallResponse(
            message=message,
            usage=UsageSnapshot(
                input_tokens=_request_input_tokens(request),
                output_tokens=estimate_tokens(message.text or "x" * 40),
            ),
            stop_reason="end_turn",
        )


class OracleStubClient:
    """Deterministic seeded navigation policy for both harness interfaces."""

    def __init__(self, skill: float = 1.0, seed: int = 0) -> None:
        self.skill = skill
        self._rng = random.Random((seed * 1_000_003) ^ int(skill * 1000))

    async def complete(self, request: ModelCallRequest) -> ModelCallResponse:
        """Dispatch on interface: tool schemas present => generic; else custom JSON."""
        if request.tools:
            message = self._generic_policy(request)
        else:
            message = ConversationMessage(
                role="assistant", content=[TextBlock(text=self._custom_policy(request))]
            )
        out_text = message.text + "".join(
            json.dumps(b.input) for b in message.content if isinstance(b, ToolUseBlock)
        )
        return ModelCallResponse(
            message=message,
            usage=UsageSnapshot(
                input_tokens=_request_input_tokens(request),
                output_tokens=estimate_tokens(out_text or "ok"),
            ),
            stop_reason="end_turn",
        )

    # ------------------------- custom-harness policy -------------------------

    def _custom_policy(self, request: ModelCallRequest) -> str:
        obs_text = request.messages[-1].text
        try:
            obs = json.loads(obs_text)
        except json.JSONDecodeError:
            return json.dumps({"action": "ESCALATE", "reason": "unreadable observation"})

        retrying = bool(obs.get("feedback"))
        competent_p = 0.85 if retrying else self.skill
        if self._rng.random() > competent_p:
            return self._corrupt_custom(obs)

        valid = obs.get("valid_actions", [])
        searches = obs.get("searches", [])
        files = obs.get("files", [])
        reads = obs.get("reads", [])
        last = obs.get("last_result", {})
        keywords = extract_keywords(obs.get("goal", ""))

        if "READ" not in valid or not files:
            done = {s.get("pattern") for s in searches}
            for kw in keywords:
                if kw not in done:
                    return json.dumps({"action": "SEARCH", "pattern": re.escape(kw)[:100]})
            return json.dumps({"action": "ESCALATE", "reason": "no matches for any keyword"})

        if "ANSWER" not in valid or not reads:
            hits = last.get("hits", []) if last.get("type") == "search" else []
            target = hits[0] if hits else {"f": files[0], "l": 1}
            return json.dumps(
                {
                    "action": "READ",
                    "path": target["f"],
                    "offset": max(0, int(target.get("l", 1)) - 3),
                    "limit": 40,
                }
            )

        answer_files = []
        for r in reads:
            if r["f"] not in answer_files:
                answer_files.append(r["f"])
        first = reads[0]
        evidence = [
            {
                "path": first["f"],
                "line_start": int(first["o"]) + 1,
                "line_end": int(first["o"]) + max(1, int(first["n"])),
            }
        ]
        return json.dumps(
            {
                "action": "ANSWER",
                "files": answer_files[:5],
                "evidence": evidence,
                "answer_text": f"Found in {', '.join(answer_files[:5])}",
                "confidence": 0.9,
            }
        )

    def _corrupt_custom(self, obs: dict) -> str:
        """Seeded mistake modes that exercise the verifier + retry localization."""
        mode = self._rng.choice(["prose", "bad_action", "ghost_file", "bad_json"])
        if mode == "prose":
            return "I think the best next step would be to look around the repository."
        if mode == "bad_action":
            return json.dumps(
                {"action": "ANSWER", "files": ["definitely/not/discovered.py"], "confidence": 1.0}
            )
        if mode == "ghost_file":
            return json.dumps({"action": "READ", "path": "src/nonexistent_module.py"})
        return '{"action": "SEARCH", "pattern": '  # truncated JSON

    # ------------------------- generic-harness policy ------------------------

    def _generic_policy(self, request: ModelCallRequest) -> ConversationMessage:
        goal = request.messages[0].text if request.messages else ""
        keywords = extract_keywords(goal)

        tool_results: list[str] = []
        n_assistant_turns = 0
        for msg in request.messages:
            if msg.role == "assistant":
                n_assistant_turns += 1
            for block in msg.content:
                if isinstance(block, ToolResultBlock):
                    tool_results.append(block.content)

        if self._rng.random() > self.skill:
            return self._corrupt_generic(keywords)

        hits = _GREP_HIT_RE.findall("\n".join(tool_results))
        searched = n_assistant_turns >= 1
        read_done = any("\t" in tr for tr in tool_results)

        if not searched or (not hits and n_assistant_turns < len(keywords)):
            kw = keywords[min(n_assistant_turns, len(keywords) - 1)]
            return ConversationMessage(
                role="assistant",
                content=[
                    TextBlock(text=f"Searching for {kw!r}."),
                    ToolUseBlock(name="grep", input={"pattern": re.escape(kw)[:100]}),
                ],
            )
        if hits and not read_done:
            path, line = hits[0]
            return ConversationMessage(
                role="assistant",
                content=[
                    TextBlock(text=f"Inspecting {path}."),
                    ToolUseBlock(
                        name="read_file",
                        input={"path": path, "offset": max(0, int(line) - 3), "limit": 40},
                    ),
                ],
            )
        if hits:
            unique = []
            for path, _ in hits:
                p = path.lstrip("./")
                if p not in unique:
                    unique.append(p)
            listing = "\n".join(f"- {p}" for p in unique[:5])
            return ConversationMessage(
                role="assistant",
                content=[TextBlock(text=f"The relevant files are:\n{listing}")],
            )
        return ConversationMessage(
            role="assistant",
            content=[TextBlock(text="I could not locate any relevant files for this request.")],
        )

    def _corrupt_generic(self, keywords: list[str]) -> ConversationMessage:
        """Mistake modes typical of weak general models in a generic harness."""
        mode = self._rng.choice(["distractor", "bad_args", "premature", "hallucinate"])
        if mode == "distractor":
            return ConversationMessage(
                role="assistant",
                content=[
                    TextBlock(text="Let me check the environment first."),
                    ToolUseBlock(name="bash", input={"command": "find . -name '*.py' | head"}),
                ],
            )
        if mode == "bad_args":
            return ConversationMessage(
                role="assistant",
                content=[
                    TextBlock(text="Searching."),
                    ToolUseBlock(name="grep", input={"glob": "**/*.py"}),  # missing 'pattern'
                ],
            )
        if mode == "premature":
            return ConversationMessage(
                role="assistant",
                content=[TextBlock(text="I could not find anything relevant in the repository.")],
            )
        return ConversationMessage(
            role="assistant",
            content=[
                TextBlock(
                    text=(
                        "The relevant file is core/main_logic.py which contains the "
                        f"{keywords[0] if keywords else 'requested'} implementation."
                    )
                )
            ],
        )
