"""Merge a LoRA adapter into its base model and save full weights.

Needed on pre-Ampere GPUs (e.g. V100): vLLM's live multi-LoRA serving requires
sm80+, so we merge the adapter offline and serve the merged model as a plain model.

Usage:
    python merge_lora.py --base Qwen/Qwen2.5-3B-Instruct \
        --adapter checkpoints/qwen2.5-3b-navsearch-lora \
        --out checkpoints/qwen2.5-3b-navsearch-merged
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    args = p.parse_args()

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:
        sys.exit(f"missing dependency: {exc.name} (install training deps first)")

    dtype = getattr(torch, args.dtype)
    print(f"[merge] loading base {args.base} ({args.dtype})")
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=dtype, low_cpu_mem_usage=True
    )
    print(f"[merge] applying adapter {args.adapter}")
    model = PeftModel.from_pretrained(base, args.adapter)
    merged = model.merge_and_unload()
    print(f"[merge] saving to {args.out}")
    merged.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base).save_pretrained(args.out)
    print("[merge] done")


if __name__ == "__main__":
    main()
