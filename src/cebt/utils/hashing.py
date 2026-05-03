"""Stable hashing utilities."""

from __future__ import annotations

import hashlib


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_id(*parts: object, prefix: str) -> str:
    digest = sha256_text("|".join(str(part) for part in parts))[:16]
    return f"{prefix}:{digest}"
