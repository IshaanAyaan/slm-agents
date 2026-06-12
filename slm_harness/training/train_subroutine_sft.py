"""Full-parameter SFT for one (subroutine, model) pair.

Phase-2 models are 135M-1.5B, so full fine-tuning is cheap and uniform across the
grid (no LoRA-rank confound). Import-guarded like train_lora.py: heavy deps load
only when training starts, and --dry-run validates data anywhere.

Usage:
    python -m slm_harness.training.train_subroutine_sft \
        --base-model HuggingFaceTB/SmolLM2-135M-Instruct \
        --train .../sft/json_repair/train.jsonl --val .../sft/json_repair/val.jsonl \
        --out .../models/json_repair/smollm2-135m [--epochs 3] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from slm_harness.training.train_lora import load_chat_jsonl


def train(args: argparse.Namespace) -> None:
    try:
        import torch
        from datasets import Dataset
        from transformers import AutoTokenizer
        from trl import SFTConfig, SFTTrainer
    except ModuleNotFoundError as exc:
        sys.exit(f"Missing training dependency: {exc.name}")

    train_rows = load_chat_jsonl(args.train, args.max_examples)
    val_rows = load_chat_jsonl(args.val) if args.val else None
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def render(row: dict) -> dict:
        return {"text": tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=False)}

    train_ds = Dataset.from_list([render(r) for r in train_rows])
    val_ds = Dataset.from_list([render(r) for r in val_rows]) if val_rows else None

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0) >= (8, 0)
    sft_config = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_seq_length=args.max_seq_len,
        dataset_text_field="text",
        packing=False,
        seed=args.seed,
        logging_steps=20,
        save_strategy="no",
        eval_strategy="epoch" if val_ds is not None else "no",
        bf16=use_bf16,
        fp16=not use_bf16 and torch.cuda.is_available(),
        # Activations dominate at bs x 2k tokens even for sub-1B models; the
        # fp32-upcast LM-head logits already eat tens of GB on long batches.
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=[],
    )
    trainer = SFTTrainer(
        model=args.base_model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )
    t0 = time.time()
    result = trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    Path(args.out, "train_meta.json").write_text(json.dumps({
        "base_model": args.base_model,
        "train_path": args.train,
        "n_train": len(train_rows),
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "max_seq_len": args.max_seq_len,
        "seed": args.seed,
        "train_seconds": round(time.time() - t0, 1),
        "train_loss": result.training_loss,
    }, indent=2), encoding="utf-8")
    print(f"[train] {args.out}: loss={result.training_loss:.4f} "
          f"in {time.time() - t0:.0f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        rows = load_chat_jsonl(args.train, args.max_examples)
        if args.val:
            load_chat_jsonl(args.val)
        print(f"DRY RUN OK: {len(rows)} train examples validated")
        return
    train(args)


if __name__ == "__main__":
    main()
