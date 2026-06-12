"""Generate all Phase-2 subroutine datasets with oracle labels.

Train/val splits come from TRAIN_REPOS; the test split comes only from held-out
TEST_REPOS, so every headline number measures generalization to unseen codebases.
Also reports rules-baseline scores on the test split, computed offline.

Usage:
    python -m slm_harness.scripts.gen_subroutine_data \
        --out slm_harness/results/subroutines/data --repos-cache data/repos \
        [--train 2000 --val 150 --test 300] [--only json_repair,...]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from slm_harness.sim.repo_pool import TEST_REPOS, TRAIN_REPOS, load_index
from slm_harness.subroutines.registry import GENERATORS, REGISTRY


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def rules_score(name: str, rows: list[dict]) -> dict:
    """Offline rules-baseline metrics on one split."""
    sub = REGISTRY[name]
    n_ok = n_abstain = 0
    for ex in rows:
        out = sub.rules_baseline(ex)
        if out is None:
            n_abstain += 1
            continue
        if sub.verify(ex, out):
            n_ok += 1
    n = len(rows)
    return {"success": n_ok / n if n else 0.0,
            "abstain": n_abstain / n if n else 0.0, "n": n}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--repos-cache", required=True)
    parser.add_argument("--train", type=int, default=2000)
    parser.add_argument("--val", type=int, default=150)
    parser.add_argument("--test", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only", default=None,
                        help="comma-separated subroutine subset")
    args = parser.parse_args()

    cache = Path(args.repos_cache)
    train_idx = [load_index(r, cache) for r in sorted(TRAIN_REPOS)]
    test_idx = [load_index(r, cache) for r in sorted(TEST_REPOS)]
    for idx in train_idx + test_idx:
        print(f"[repo] {idx.repo_id}: {len(idx.files)} files, "
              f"{len(idx.symbols)} unique symbols ({idx.split})")

    names = sorted(GENERATORS)
    if args.only:
        names = [n for n in args.only.split(",") if n]

    out_root = Path(args.out)
    summary: dict[str, dict] = {}
    for name in names:
        gen = GENERATORS[name]
        t0 = time.time()
        # Val gets a different seed stream than train; both use train repos only.
        train_rows = gen(train_idx, "train", args.train, args.seed)
        val_rows = gen(train_idx, "val", args.val, args.seed + 1)
        test_rows = gen(test_idx, "test", args.test, args.seed + 2)
        # Symbol-keyed subroutines must not leak train symbols into val.
        train_syms = {r["meta"].get("symbol") for r in train_rows} - {None}
        if train_syms:
            val_rows = [r for r in val_rows
                        if r["meta"].get("symbol") not in train_syms] or val_rows[:1]
        for split, rows in (("train", train_rows), ("val", val_rows),
                            ("test", test_rows)):
            write_jsonl(out_root / name / f"{split}.jsonl", rows)
        baseline = rules_score(name, test_rows)
        summary[name] = {
            "train": len(train_rows), "val": len(val_rows), "test": len(test_rows),
            "rules_test": baseline, "gen_seconds": round(time.time() - t0, 1),
        }
        print(f"[data] {name}: train={len(train_rows)} val={len(val_rows)} "
              f"test={len(test_rows)} rules_test={baseline['success']:.3f} "
              f"(abstain {baseline['abstain']:.3f}) "
              f"in {summary[name]['gen_seconds']}s")

    (out_root / "datagen_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] wrote {out_root}/datagen_summary.json")


if __name__ == "__main__":
    main()
