from __future__ import annotations

import csv

import numpy as np

from cebt.analysis.tables import make_tables
from cebt.evaluation.bootstrap import clustered_bootstrap_ci, leave_one_group_out
from cebt.utils.io import write_json, write_jsonl


def test_clustered_bootstrap_reports_group_count() -> None:
    values = np.asarray([[1.0, 1.5], [2.0, 2.5], [3.0, 3.5], [4.0, 4.5]])
    groups = np.asarray(["a", "a", "b", "b"], dtype=object)

    interval = clustered_bootstrap_ci(
        values,
        groups,
        lambda rows: float(np.mean(rows[:, 1] - rows[:, 0])),
        n_boot=50,
        seed=3,
    )

    assert interval["groups"] == 2
    assert interval["mean"] == 0.5
    assert interval["lo"] <= interval["mean"] <= interval["hi"]


def test_leave_one_group_out_reports_extreme_groups() -> None:
    values = np.asarray([[1.0], [2.0], [3.0], [100.0]])
    groups = np.asarray(["a", "a", "b", "c"], dtype=object)

    result = leave_one_group_out(values, groups, lambda rows: float(np.mean(rows[:, 0])))

    assert result["groups"] == 3
    assert result["full"] == 26.5
    assert result["min_group"] == "c"
    assert result["max_group"] == "a"
    assert result["min"] < result["full"] < result["max"]


def test_make_tables_writes_paired_rank_ic_table(tmp_path) -> None:
    metadata = {
        "mse": 0.1,
        "mse_ci": {"mean": 0.1, "lo": 0.09, "hi": 0.11},
        "abnormal_return_rank_ic": 0.5,
        "rank_ic_ci": {"mean": 0.5, "lo": 0.1, "hi": 0.8},
    }
    write_json(tmp_path / "cebt_eval_metrics.json", metadata)
    write_json(tmp_path / "baseline_eval_metrics.json", {**metadata, "mse": 0.2})
    targets = [-0.2, -0.1, 0.1, 0.2]
    write_jsonl(
        tmp_path / "cebt_predictions.jsonl",
        [
            {
                "sample_id": f"s{idx}",
                "prediction_abnormal_return": target,
                "prediction_volatility_jump": 0.0,
                "prediction_volume_jump": 0.0,
                "target_abnormal_return": target,
                "target_volatility_jump": 0.0,
                "target_volume_jump": 0.0,
            }
            for idx, target in enumerate(targets)
        ],
    )
    write_jsonl(
        tmp_path / "baseline_predictions.jsonl",
        [
            {
                "sample_id": f"s{idx}",
                "prediction_abnormal_return": -target,
                "prediction_volatility_jump": 0.0,
                "prediction_volume_jump": 0.0,
                "target_abnormal_return": target,
                "target_volatility_jump": 0.0,
                "target_volume_jump": 0.0,
            }
            for idx, target in enumerate(targets)
        ],
    )

    summary = make_tables(tmp_path)

    assert summary["paired_rank_rows"] == 1
    with (tmp_path / "table_paired_rank_ic_comparisons.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["baseline_model"] == "baseline"
    assert float(rows[0]["reference_minus_baseline_rank_ic"]) > 0.0
