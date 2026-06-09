"""Harness adapters: generic (ohmo-style) and custom (co-designed narrow harness)."""

from research.slm_harness.harness.custom import CustomSearchHarness
from research.slm_harness.harness.generic import GenericHarness
from research.slm_harness.harness.state import CustomHarnessState, HarnessOutcome

__all__ = ["CustomHarnessState", "CustomSearchHarness", "GenericHarness", "HarnessOutcome"]
