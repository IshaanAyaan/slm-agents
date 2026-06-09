"""Application configuration loading."""

import json
import os

DEFAULT_RETRY_COUNT = 3
CONFIG_ENV_VAR = "ORDERFLOW_CONFIG"


def load_config(path: str | None = None) -> dict:
    """Load JSON config from the given path or the env-var location."""
    location = path or os.environ.get(CONFIG_ENV_VAR, "config.json")
    if not os.path.exists(location):
        return {"retries": DEFAULT_RETRY_COUNT}
    with open(location, "r", encoding="utf-8") as fh:
        return json.load(fh)
