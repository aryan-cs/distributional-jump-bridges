"""Config loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    config = load_yaml(config_path)
    model_path = config.get("model", {}).get("config_path")
    if model_path:
        model_config_path = Path(model_path)
        if not model_config_path.is_absolute():
            model_config_path = (config_path.parents[1] / model_path).resolve()
            if not model_config_path.exists():
                model_config_path = (Path.cwd() / model_path).resolve()
        model_config = load_yaml(model_config_path)
        config = deep_merge(config, model_config)
    return config


def ensure_dir(path: str | Path) -> Path:
    value = Path(path)
    value.mkdir(parents=True, exist_ok=True)
    return value
