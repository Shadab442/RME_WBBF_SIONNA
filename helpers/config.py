"""Shared config.yaml loader.

Each script reads only `common` merged with its own section (its own values
override `common`) -- see config.yaml's own header comment for the schema.
"""

import os

import yaml

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")


def load_config(script_name: str = None) -> dict:
    """:param script_name: e.g. "test_dynamic_scenario_tilts_effect". If
    None, returns `common` alone."""
    with open(_CONFIG_PATH) as f:
        full_config = yaml.safe_load(f)
    common = full_config.get("common", {})
    if script_name is None:
        return common
    return {**common, **full_config.get(script_name, {})}
