"""Export subroutine datasets as 3-turn chat JSONL for SFT.

Reuses the Phase-1 chat format (system/user/assistant) so the same validation and
training stack applies. The assistant turn is always compact strict JSON.

Usage:
    python -m slm_harness.training.build_subroutine_sft \
        --data slm_harness/results/subroutines/data --out .../sft [--only a,b]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_harness.subroutines.registry import REGISTRY


def export_split(name: str, data_dir: Path, out_dir: Path, split: str) -> int:
    sub = REGISTRY[name]
    src = data_dir / name / f"{split}.jsonl"
    if not src.is_file():
        return 0
    rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{split}.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for ex in rows:
            chat = sub.to_chat(ex)
            # The gold assistant turn must round-trip through the subroutine's own
            # parser and verifier; a failure here is a generator bug, not noise.
            payload, err = sub.parse_output(chat["messages"][2]["content"])
            if payload is None or not sub.verify(ex, payload):
                raise ValueError(f"{name}/{split}/{ex['id']}: gold fails own "
                                 f"verifier ({err or 'verify=False'})")
            fh.write(json.dumps(chat, ensure_ascii=False) + "\n")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--only", default=None)
    args = parser.parse_args()

    data_dir, out_root = Path(args.data), Path(args.out)
    names = sorted(REGISTRY)
    if args.only:
        names = [n for n in args.only.split(",") if n]
    for name in names:
        counts = {s: export_split(name, data_dir, out_root / name, s)
                  for s in ("train", "val", "test")}
        print(f"[sft] {name}: {counts}")


if __name__ == "__main__":
    main()
