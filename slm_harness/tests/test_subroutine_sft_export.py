"""SFT export produces valid 3-turn chat JSONL the Phase-1 loader accepts."""

from __future__ import annotations

import json
from pathlib import Path

from slm_harness.subroutines.registry import GENERATORS
from slm_harness.training.build_subroutine_sft import export_split
from slm_harness.training.train_lora import load_chat_jsonl


def test_export_round_trip(mini_index, tmp_path: Path):
    name = "action_router"
    rows = GENERATORS[name]([mini_index], "train", 5, seed=0)
    data_dir = tmp_path / "data"
    (data_dir / name).mkdir(parents=True)
    with (data_dir / name / "train.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    out_dir = tmp_path / "sft" / name
    n = export_split(name, data_dir, out_dir, "train")
    assert n == 5
    loaded = load_chat_jsonl(str(out_dir / "train.jsonl"))
    assert len(loaded) == 5
    for row in loaded:
        json.loads(row["messages"][2]["content"])


def test_export_missing_split_is_zero(tmp_path: Path):
    assert export_split("json_repair", tmp_path, tmp_path / "out", "val") == 0
