"""Shared script helpers."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from cebt.data.sec import SECClient
from cebt.utils.config import ensure_dir, load_config


def parse_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model-name", default="cebt")
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else repo_root() / value


def load_run_config(config_path: str | Path) -> dict[str, Any]:
    return load_config(resolve_path(config_path))


def output_dir(args: argparse.Namespace, default: str = "data/processed/pilot") -> Path:
    return ensure_dir(resolve_path(args.output_dir or default))


def processed_dir(config: dict[str, Any]) -> Path:
    run_name = config.get("project", {}).get("run_name", "pilot")
    return resolve_path(Path("data/processed") / str(run_name))


def sec_client(config: dict[str, Any]) -> SECClient:
    sec = config.get("sec", {})
    user_agent_env = sec.get("user_agent_env", "SEC_USER_AGENT")
    user_agent = os.getenv(user_agent_env) or os.getenv("SEC_USER_AGENT")
    if not user_agent:
        raise SystemExit(
            f"Set {user_agent_env} to a descriptive SEC User-Agent with contact information."
        )
    return SECClient(
        user_agent=user_agent,
        base_url=sec.get("base_url", "https://data.sec.gov"),
        archives_url=sec.get("archives_url", "https://www.sec.gov/Archives/edgar/data"),
        ticker_url=sec.get("ticker_url", "https://www.sec.gov/files/company_tickers.json"),
        max_requests_per_second=float(sec.get("max_requests_per_second", 8.0)),
    )
