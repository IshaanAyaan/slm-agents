"""LoRA SFT training interface for the navigation-subagent SLM.

Deliberately import-guarded: the heavy dependencies (torch/transformers/peft/trl)
are only imported when training actually starts, so this module is importable and
`--dry-run` works in any environment. No credentials or weights are bundled.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    """All LoRA training hyperparameters."""

    base_model: str
    train_path: str
    val_path: str | None
    out_dir: str
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    epochs: float = 3.0
    lr: float = 1e-4
    batch_size: int = 8
    grad_accum: int = 2
    max_seq_len: int = 2048
    max_examples: int | None = None
    seed: int = 0


def load_chat_jsonl(path: str, max_examples: int | None = None) -> list[dict]:
    """Load and validate the distillation chat JSONL."""
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if max_examples is not None and len(rows) >= max_examples:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            msgs = row.get("messages")
            if not (
                isinstance(msgs, list)
                and len(msgs) == 3
                and [m.get("role") for m in msgs] == ["system", "user", "assistant"]
            ):
                raise ValueError(f"{path}:{i + 1}: not a valid 3-turn chat example")
            json.loads(msgs[2]["content"])  # assistant turn must be valid action JSON
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no examples")
    return rows


def train(config: TrainConfig) -> None:
    """Run LoRA SFT (requires torch/transformers/peft/trl/datasets installed)."""
    try:
        import torch  # noqa: F401
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoTokenizer
        from trl import SFTConfig, SFTTrainer
    except ModuleNotFoundError as exc:
        sys.exit(
            f"Missing training dependency: {exc.name}. "
            "See research/slm_harness/training/README.md for the install command."
        )

    train_rows = load_chat_jsonl(config.train_path, config.max_examples)
    val_rows = load_chat_jsonl(config.val_path) if config.val_path else None
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)

    def render(row: dict) -> dict:
        return {
            "text": tokenizer.apply_chat_template(
                row["messages"], tokenize=False, add_generation_prompt=False
            )
        }

    train_ds = Dataset.from_list([render(r) for r in train_rows])
    val_ds = Dataset.from_list([render(r) for r in val_rows]) if val_rows else None

    peft_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    # bf16 needs Ampere (sm80+); fall back to fp16 on older GPUs like V100 (sm70).
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0) >= (8, 0)
    sft_config = SFTConfig(
        output_dir=config.out_dir,
        num_train_epochs=config.epochs,
        learning_rate=config.lr,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accum,
        max_seq_length=config.max_seq_len,
        seed=config.seed,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if val_ds is not None else "no",
        bf16=use_bf16,
        fp16=not use_bf16,
        report_to=[],
    )
    trainer = SFTTrainer(
        model=config.base_model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(config.out_dir)
    Path(config.out_dir, "train_config.json").write_text(
        json.dumps(config.__dict__, indent=2), encoding="utf-8"
    )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data + config only; no model download, no GPU required.",
    )
    args = parser.parse_args()

    config = TrainConfig(
        base_model=args.base_model,
        train_path=args.train,
        val_path=args.val,
        out_dir=args.out,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        max_examples=args.max_examples,
        seed=args.seed,
    )
    if args.dry_run:
        rows = load_chat_jsonl(config.train_path, config.max_examples)
        if config.val_path:
            load_chat_jsonl(config.val_path)
        print(f"DRY RUN OK: {len(rows)} train examples validated; config: {config}")
        return
    train(config)


if __name__ == "__main__":
    main()
