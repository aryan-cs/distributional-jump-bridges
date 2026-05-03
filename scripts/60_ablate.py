"""Run CEBT baseline ablations."""

from __future__ import annotations

from copy import deepcopy

from cebt.cli import load_run_config, output_dir, parse_args, resolve_path
from cebt.evaluation.evaluate import evaluate_model
from cebt.training.train import train_model
from cebt.utils.io import write_json


def main() -> None:
    args = parse_args("Run CEBT ablations")
    config = load_run_config(args.config)
    run_dir = output_dir(args, "data/runs/pilot")
    feature_path = resolve_path("data/processed/pilot/features.npz")
    metadata_path = resolve_path("data/processed/pilot/features_metadata.jsonl")
    results = {}
    specs = [
        ("no_event", config),
        ("text_only", config),
        ("concat", config),
        ("cebt_no_controls", _without_control_loss(config)),
        ("cebt", config),
    ]
    for model_name, model_config in specs:
        train_model(model_config, feature_path, run_dir, model_name=model_name)
        results[model_name] = evaluate_model(
            model_config,
            feature_path,
            metadata_path,
            run_dir / f"{model_name}.pt",
            run_dir,
        )
    write_json(run_dir / "ablation_summary.json", results)
    print(results)


def _without_control_loss(config: dict) -> dict:
    modified = deepcopy(config)
    modified.setdefault("loss", {})["control_delta_weight"] = 0.0
    modified["loss"]["consistency_weight"] = 0.0
    return modified


if __name__ == "__main__":
    main()
