from __future__ import annotations

import pytest

from cebt.evaluation.leakage import validate_feature_metadata
from cebt.utils.io import write_jsonl


def test_evaluation_refuses_leaky_feature_metadata(tmp_path) -> None:
    metadata = tmp_path / "metadata.jsonl"
    write_jsonl(
        metadata,
        [
            {
                "sample_id": "leaky",
                "feature_max_date": "2025-01-04",
                "label_start_date": "2025-01-03",
                "label_source": "future_returns_only",
            }
        ],
    )
    with pytest.raises(ValueError, match="Feature leakage"):
        validate_feature_metadata(metadata)
