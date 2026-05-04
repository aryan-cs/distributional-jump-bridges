"""Fast code-path smoke test for DJB research modules."""

from __future__ import annotations

import numpy as np
import torch

from cebt.cli import load_run_config, output_dir, parse_args
from cebt.models.cebt import ModelConfig, build_model
from cebt.training.losses import LossWeights, cebt_loss
from cebt.utils.io import write_json


def main() -> None:
    args = parse_args("Run DJB code-path smoke test")
    config = load_run_config(args.config)
    out = output_dir(args, "data/runs/smoke")
    model_config = ModelConfig.from_dict(config.get("model", {}))
    model = build_model(args.model_name, model_config)
    batch = {
        "x_pre": torch.randn(4, 8, model_config.price_features),
        "event_embedding": torch.randn(4, model_config.event_embedding_dim),
        "metadata": torch.randn(4, model_config.metadata_features),
        "y": torch.randn(4, model_config.output_dim),
        "is_event": torch.tensor([1.0, 1.0, 0.0, 0.0]),
    }
    outputs = model(batch["x_pre"], batch["event_embedding"], batch["metadata"])
    loss, metrics = cebt_loss(
        outputs,
        batch["y"],
        batch["is_event"],
        LossWeights.from_dict(config.get("loss", {})),
    )
    summary = {
        "model_name": args.model_name,
        "loss": float(loss.detach()),
        "metrics": metrics,
        "prediction_shape": list(outputs["prediction"].shape),
        "finite": bool(np.isfinite(float(loss.detach()))),
        "note": "Code-path smoke only; not a research result.",
    }
    write_json(out / "smoke_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
