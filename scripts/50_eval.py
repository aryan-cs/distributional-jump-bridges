"""Evaluate CEBT or baseline checkpoints."""

from __future__ import annotations

from cebt.cli import load_run_config, output_dir, parse_args, processed_dir
from cebt.evaluation.evaluate import evaluate_model


def main() -> None:
    args = parse_args("Evaluate CEBT")
    config = load_run_config(args.config)
    run_dir = output_dir(args, "data/runs/pilot")
    features = processed_dir(config)
    feature_path = features / "features.npz"
    metadata_path = features / "features_metadata.jsonl"
    checkpoint_path = run_dir / f"{args.model_name}.pt"
    metrics = evaluate_model(
        config,
        feature_path,
        metadata_path,
        checkpoint_path,
        run_dir,
        intervention=args.intervention,
    )
    print(metrics)


if __name__ == "__main__":
    main()
