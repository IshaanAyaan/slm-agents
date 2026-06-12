"""Evaluate (subroutine x model x condition) on the held-out test split.

Conditions:
  ft    fine-tuned specialist checkpoint (models_root/<subroutine>/<size>)
  base  the un-fine-tuned instruct model under the identical harness prompt

Protocol: greedy decoding; schema-invalid replies get exactly one retry with the
parse error fed back (the harness's retry-localization principle). Success is the
subroutine's deterministic verifier; nothing is judged by a model.

Usage (GPU):
    python -m slm_harness.evals.subroutine_eval \
        --data .../subroutines/data --models-root .../subroutines/models \
        --out .../subroutines/eval --sizes smollm2-135m,smollm2-360m \
        --conditions ft,base [--subroutines a,b] [--max-test 300]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from slm_harness.subroutines.registry import MODEL_GRID, REGISTRY

MAX_NEW_TOKENS = {
    "json_repair": 220,
    "action_router": 24,
    "path_normalizer": 48,
    "search_query_gen": 40,
    "search_hit_ranker": 16,
    "read_span_selector": 28,
    "evidence_judge": 48,
    "trace_localizer": 64,
}
RETRY_SUFFIX = ("Your previous reply was invalid: {err}. "
                "Reply again with ONLY the corrected strict-JSON object.")


def load_examples(data_dir: Path, name: str, max_test: int | None) -> list[dict]:
    rows = [json.loads(l) for l in
            (data_dir / name / "test.jsonl").read_text(encoding="utf-8").splitlines()
            if l]
    return rows[:max_test] if max_test else rows


class HFRunner:
    """Batched greedy generation for one loaded model."""

    def __init__(self, model_path: str, batch_size: int):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype,
            device_map="cuda" if torch.cuda.is_available() else None)
        self.model.eval()
        self.batch_size = batch_size

    def close(self) -> None:
        del self.model
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def generate(self, chats: list[list[dict]], max_new_tokens: int) -> tuple[list[str], dict]:
        """Greedy-decode a list of chat prompts; returns texts + usage stats."""
        torch = self.torch
        texts: list[str] = []
        tokens_in = tokens_out = 0
        t0 = time.time()
        prompts = [self.tokenizer.apply_chat_template(
            c, tokenize=False, add_generation_prompt=True) for c in chats]
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i:i + self.batch_size]
            enc = self.tokenizer(batch, return_tensors="pt", padding=True,
                                 truncation=True, max_length=4096)
            if torch.cuda.is_available():
                enc = {k: v.to("cuda") for k, v in enc.items()}
            with torch.no_grad():
                out = self.model.generate(
                    **enc, max_new_tokens=max_new_tokens, do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id)
            gen = out[:, enc["input_ids"].shape[1]:]
            texts.extend(self.tokenizer.batch_decode(gen, skip_special_tokens=True))
            tokens_in += int(enc["attention_mask"].sum())
            tokens_out += int((gen != self.tokenizer.pad_token_id).sum())
        return texts, {"tokens_in": tokens_in, "tokens_out": tokens_out,
                       "wall_seconds": round(time.time() - t0, 2)}


def eval_subroutine(runner: HFRunner, name: str, examples: list[dict]) -> dict:
    """Score one subroutine: pass 1 greedy + one schema-retry pass."""
    sub = REGISTRY[name]
    max_new = MAX_NEW_TOKENS[name]
    chats = [[{"role": "system", "content": sub.system_prompt()},
              {"role": "user", "content": sub.render_user(ex)}] for ex in examples]
    replies, usage = runner.generate(chats, max_new)
    records = [sub.score(ex, raw) for ex, raw in zip(examples, replies)]

    # One retry for schema-invalid replies only (verifier failures are final).
    retry_ids = [i for i, r in enumerate(records) if not r["schema_valid"]]
    retry_usage = {"tokens_in": 0, "tokens_out": 0, "wall_seconds": 0.0}
    if retry_ids:
        retry_chats = []
        for i in retry_ids:
            retry_chats.append(chats[i] + [
                {"role": "assistant", "content": replies[i]},
                {"role": "user", "content": RETRY_SUFFIX.format(err=records[i]["error"])},
            ])
        retry_replies, retry_usage = runner.generate(retry_chats, max_new)
        for i, raw in zip(retry_ids, retry_replies):
            records[i]["retry"] = sub.score(examples[i], raw)

    n = len(records)
    n_valid = sum(r["schema_valid"] for r in records)
    n_ok = sum(r["success"] for r in records)
    n_ok_retry = sum(
        r["success"] or r.get("retry", {}).get("success", False) for r in records)
    return {
        "subroutine": name,
        "n": n,
        "schema_valid": n_valid / n,
        "success": n_ok / n,
        "success_retry": n_ok_retry / n,
        "usage": {k: usage[k] + retry_usage[k] for k in usage},
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--models-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sizes", required=True)
    parser.add_argument("--conditions", default="ft,base")
    parser.add_argument("--subroutines", default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    data_dir, out_dir = Path(args.data), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = [s for s in args.sizes.split(",") if s]
    conditions = [c for c in args.conditions.split(",") if c]
    names = ([n for n in args.subroutines.split(",") if n]
             if args.subroutines else sorted(REGISTRY))

    for size in sizes:
        hf_id = MODEL_GRID[size]["hf_id"]
        for cond in conditions:
            for name in names:
                out_path = out_dir / f"{name}__{size}__{cond}.json"
                if out_path.is_file():
                    print(f"[skip] {out_path.name} exists")
                    continue
                model_path = (hf_id if cond == "base"
                              else str(Path(args.models_root) / name / size))
                if cond == "ft" and not Path(model_path, "config.json").is_file():
                    print(f"[warn] missing checkpoint {model_path}; skipping")
                    continue
                examples = load_examples(data_dir, name, args.max_test)
                t0 = time.time()
                runner = HFRunner(model_path, args.batch_size)
                result = eval_subroutine(runner, name, examples)
                runner.close()
                result.update({"size": size, "condition": cond,
                               "model_path": model_path,
                               "params_m": MODEL_GRID[size]["params_m"],
                               "total_seconds": round(time.time() - t0, 1)})
                out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
                print(f"[eval] {name} {size} {cond}: "
                      f"success={result['success']:.3f} "
                      f"retry={result['success_retry']:.3f} "
                      f"valid={result['schema_valid']:.3f} "
                      f"({result['total_seconds']}s)")


if __name__ == "__main__":
    main()
