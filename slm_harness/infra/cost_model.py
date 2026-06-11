"""Self-hosted cost model: turn measured GPU throughput into per-token pricing.

We never pay per API token here. Instead, the dollar cost of a served model is its
*amortized GPU cost while it is doing work*. For a model served on ``num_gpus`` at a
known $/GPU-hour, with empirically measured prefill (prompt) and decode (generation)
throughput on THIS hardware, the marginal cost of a token is the GPU-time it occupies:

    gpu_cost_per_second   = num_gpus * gpu_hourly_usd / 3600
    usd_per_input_token   = gpu_cost_per_second / prompt_tok_s
    usd_per_output_token  = gpu_cost_per_second / gen_tok_s

Bigger models (the 70B teacher) are correctly more expensive per token: they need more
GPUs AND run fewer tokens/sec. Small students are cheap because one GPU serves them
fast. This makes cost-per-successful-task a fair, hardware-grounded comparison with no
paid API anywhere in the loop.

Throughput is measured, not guessed: ``measure_endpoint`` benchmarks a live vLLM
endpoint and reports realized prompt/gen tok/s, which ``selfhost_pricing`` converts to a
``PricingConfig``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass

from slm_harness import _bootstrap  # noqa: F401

from slm_harness.schemas.config import PricingConfig

# Documented default; override with the real amortized/cloud rate for your H100s.
DEFAULT_H100_HOURLY_USD = 2.50


@dataclass
class ThroughputMeasurement:
    """Realized serving throughput for one model on the target hardware."""

    model: str
    num_gpus: int
    prompt_tok_s: float
    gen_tok_s: float
    gpu_hourly_usd: float
    samples: int = 0


def selfhost_pricing(
    *,
    num_gpus: int,
    gpu_hourly_usd: float,
    prompt_tok_s: float,
    gen_tok_s: float,
) -> PricingConfig:
    """Convert measured GPU throughput into a per-million-token PricingConfig."""
    if prompt_tok_s <= 0 or gen_tok_s <= 0:
        raise ValueError("throughput must be positive")
    gpu_cost_per_s = num_gpus * gpu_hourly_usd / 3600.0
    return PricingConfig(
        usd_per_million_input=gpu_cost_per_s / prompt_tok_s * 1_000_000.0,
        usd_per_million_output=gpu_cost_per_s / gen_tok_s * 1_000_000.0,
        is_placeholder=False,  # grounded in measured hardware throughput
    )


def pricing_from_measurement(m: ThroughputMeasurement) -> PricingConfig:
    """PricingConfig from a ThroughputMeasurement."""
    return selfhost_pricing(
        num_gpus=m.num_gpus,
        gpu_hourly_usd=m.gpu_hourly_usd,
        prompt_tok_s=m.prompt_tok_s,
        gen_tok_s=m.gen_tok_s,
    )


async def measure_endpoint(
    *,
    base_url: str,
    model: str,
    num_gpus: int,
    gpu_hourly_usd: float = DEFAULT_H100_HOURLY_USD,
    api_key: str = "EMPTY",
    n_requests: int = 16,
    prompt_tokens_target: int = 800,
    max_output_tokens: int = 128,
) -> ThroughputMeasurement:
    """Benchmark a live OpenAI-compatible endpoint; return realized throughput.

    Sends concurrent requests with a representative prompt size and measures
    aggregate prompt and generation tokens per wall-clock second.
    """
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    filler = ("the quick brown fox jumps over the lazy dog. " * 200)[: prompt_tokens_target * 4]
    prompt = f"Summarize the following text in one word.\n\n{filler}"

    async def _one() -> tuple[int, int]:
        r = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_output_tokens,
            temperature=0.0,
        )
        u = r.usage
        return int(getattr(u, "prompt_tokens", 0) or 0), int(getattr(u, "completion_tokens", 0) or 0)

    t0 = time.monotonic()
    results = await asyncio.gather(*[_one() for _ in range(n_requests)])
    elapsed = max(1e-6, time.monotonic() - t0)
    in_tok = sum(r[0] for r in results)
    out_tok = sum(r[1] for r in results)
    return ThroughputMeasurement(
        model=model,
        num_gpus=num_gpus,
        prompt_tok_s=in_tok / elapsed,
        gen_tok_s=out_tok / elapsed,
        gpu_hourly_usd=gpu_hourly_usd,
        samples=n_requests,
    )


def main() -> None:
    """CLI: measure an endpoint and emit a pricing JSON fragment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-gpus", type=int, required=True)
    parser.add_argument("--gpu-hourly-usd", type=float, default=DEFAULT_H100_HOURLY_USD)
    parser.add_argument("--n-requests", type=int, default=16)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    m = asyncio.run(
        measure_endpoint(
            base_url=args.base_url,
            model=args.model,
            num_gpus=args.num_gpus,
            gpu_hourly_usd=args.gpu_hourly_usd,
            n_requests=args.n_requests,
        )
    )
    pricing = pricing_from_measurement(m)
    payload = {
        "model": m.model,
        "num_gpus": m.num_gpus,
        "gpu_hourly_usd": m.gpu_hourly_usd,
        "prompt_tok_s": round(m.prompt_tok_s, 1),
        "gen_tok_s": round(m.gen_tok_s, 1),
        "pricing": pricing.model_dump(),
    }
    text = json.dumps(payload, indent=2)
    print(text)
    if args.out:
        from pathlib import Path

        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
