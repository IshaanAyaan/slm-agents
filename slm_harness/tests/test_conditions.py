"""C1-C6 factorial structure."""

from __future__ import annotations

from slm_harness.conditions import (
    default_conditions,
    default_harness_profiles,
    default_model_profiles,
)


def test_six_conditions_cover_factorial() -> None:
    models = default_model_profiles()
    conditions = default_conditions(models)
    assert len(conditions) == 6
    cells = {(c.model_factor, c.harness_factor) for c in conditions.values()}
    assert cells == {
        ("large_general", "generic"),
        ("small_general", "generic"),
        ("small_general", "custom"),
        ("small_finetuned", "generic"),
        ("small_finetuned", "custom"),
        ("large_general", "custom"),
    }


def test_condition_assignments() -> None:
    models = default_model_profiles()
    c = default_conditions(models)
    assert (c["C1"].model_factor, c["C1"].harness_factor) == ("large_general", "generic")
    assert (c["C2"].model_factor, c["C2"].harness_factor) == ("small_general", "generic")
    assert (c["C3"].model_factor, c["C3"].harness_factor) == ("small_general", "custom")
    assert (c["C4"].model_factor, c["C4"].harness_factor) == ("small_finetuned", "generic")
    assert (c["C5"].model_factor, c["C5"].harness_factor) == ("small_finetuned", "custom")
    assert (c["C6"].model_factor, c["C6"].harness_factor) == ("large_general", "custom")


def test_harness_profiles() -> None:
    profiles = default_harness_profiles()
    assert profiles["generic"].kind == "generic"
    assert profiles["custom"].kind == "custom"
    assert profiles["custom"].max_retries_per_step >= 1
