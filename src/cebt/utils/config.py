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
    config = _load_config_with_extends(config_path)
    model_path = config.get("model", {}).get("config_path")
    if model_path:
        model_config_path = _resolve_reference(model_path, config_path)
        model_config = load_yaml(model_config_path)
        config = deep_merge(config, model_config)
    return config


def _load_config_with_extends(config_path: Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    extends_path = config.pop("extends", None)
    if not extends_path:
        return config
    base_config = _load_config_with_extends(_resolve_reference(extends_path, config_path))
    return deep_merge(base_config, config)


def _resolve_reference(path: str | Path, config_path: Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    candidates = [
        (config_path.parent / value).resolve(),
        (config_path.parents[1] / value).resolve(),
        (Path.cwd() / value).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def ensure_dir(path: str | Path) -> Path:
    value = Path(path)
    value.mkdir(parents=True, exist_ok=True)
    return value
