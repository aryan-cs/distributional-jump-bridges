"""Generate compact result tables."""

from __future__ import annotations

from pathlib import Path

from cebt.utils.io import read_json, write_csv, write_json


def make_tables(run_dir: str | Path) -> dict:
    root = Path(run_dir)
    metric_rows = []
    for path in sorted(root.glob("*_eval_metrics.json")):
        metrics = read_json(path)
        model = path.name.replace("_eval_metrics.json", "")
        for key, value in metrics.items():
            metric_rows.append({"model": model, "metric": key, "value": value})
    write_csv(root / "table_eval_metrics.csv", metric_rows)
    summary = {"metric_rows": len(metric_rows), "source_dir": str(root)}
    write_json(root / "tables_summary.json", summary)
    return summary
