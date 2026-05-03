"""Train CEBT or baseline models."""

from __future__ import annotations

from cebt.cli import load_run_config, output_dir, parse_args, processed_dir
from cebt.training.train import train_model


def main() -> None:
    args = parse_args("Train CEBT")
    config = load_run_config(args.config)
    run_dir = output_dir(args, "data/runs/pilot")
    feature_path = processed_dir(config) / "features.npz"
    summary = train_model(config, feature_path, run_dir, model_name=args.model_name)
    print(summary)


if __name__ == "__main__":
    main()
