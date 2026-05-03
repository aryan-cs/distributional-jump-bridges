"""Evaluate CEBT or baseline checkpoints."""

from __future__ import annotations

from cebt.cli import load_run_config, output_dir, parse_args, resolve_path
from cebt.evaluation.evaluate import evaluate_model


def main() -> None:
    args = parse_args("Evaluate CEBT")
    config = load_run_config(args.config)
    run_dir = output_dir(args, "data/runs/pilot")
    feature_path = resolve_path("data/processed/pilot/features.npz")
    metadata_path = resolve_path("data/processed/pilot/features_metadata.jsonl")
    checkpoint_path = run_dir / f"{args.model_name}.pt"
    metrics = evaluate_model(config, feature_path, metadata_path, checkpoint_path, run_dir)
    print(metrics)


if __name__ == "__main__":
    main()
