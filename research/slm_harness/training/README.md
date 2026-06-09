# LoRA fine-tuning interface (not executed in this repo by default)

This directory defines the training side of the distillation pipeline. **No training
is run automatically** — it requires a GPU machine and locally downloaded model
weights. Everything below is a clean, reproducible interface to run when ready.

## Model candidates

| Priority | Model | HF id | Notes |
|---|---|---|---|
| primary | Qwen3-4B (instruct) | `Qwen/Qwen3-4B-Instruct-2507` | best quality/size tradeoff |
| primary (cheaper) | Qwen3-1.7B | `Qwen/Qwen3-1.7B` | for the data-efficiency / size ablation |
| secondary | Llama-3.2-3B | `meta-llama/Llama-3.2-3B-Instruct` | license-gated; cross-family check |

## Data

Produced by the distillation builder (chat-format JSONL, one example per step):

```bash
python research/slm_harness/scripts/build_distillation.py \
  --runs <results>/runs.jsonl \
  --tasks research/slm_harness/tasks/<suite>/<suite>.json \
  --out research/slm_harness/results/distillation
# -> train.jsonl / val.jsonl / test.jsonl + stats.json
```

Each line: `{"messages":[{system},{user: observation JSON},{assistant: action JSON}], "meta":{...}}`
— directly consumable by TRL `SFTTrainer`, axolotl, or LLaMA-Factory.

## Environment

```bash
pip install "torch>=2.3" "transformers>=4.51" "peft>=0.11" "trl>=0.9" "datasets>=2.19" \
    "accelerate>=0.30" bitsandbytes
```

## Train (TRL + PEFT, LoRA rank 32–64)

```bash
python research/slm_harness/training/train_lora.py \
  --base-model Qwen/Qwen3-4B-Instruct-2507 \
  --train research/slm_harness/results/distillation/train.jsonl \
  --val research/slm_harness/results/distillation/val.jsonl \
  --out research/slm_harness/training/checkpoints/qwen3-4b-navsearch-lora \
  --lora-rank 64 --lora-alpha 128 --epochs 3 --lr 1e-4 --batch-size 8
```

Use `--dry-run` to validate data/config without any GPU or model download.

For the **data-efficiency curve** (Figure 4), train on nested subsets and evaluate
each checkpoint with the C5 condition:

```bash
for n in 100 250 500 1000 2000; do
  python research/slm_harness/training/train_lora.py ... --max-examples $n \
    --out .../qwen3-4b-navsearch-lora-n$n
done
```

## Serve for evaluation (C4/C5)

```bash
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --enable-lora --lora-modules navsearch=research/slm_harness/training/checkpoints/qwen3-4b-navsearch-lora \
  --port 8000
```

Then set the `small_finetuned` model profile to `provider="openai_compat"`,
`base_url="http://localhost:8000/v1"`, `model_name="navsearch"`.

## Record the fine-tuning cost

Whatever you spend (GPU-hours × rate) goes into `ExperimentConfig.finetune_cost_usd`
so the break-even analysis uses the real number.
