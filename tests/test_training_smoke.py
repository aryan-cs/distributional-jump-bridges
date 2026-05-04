from __future__ import annotations

import numpy as np

from cebt.evaluation.evaluate import evaluate_model
from cebt.training.train import train_model
from cebt.utils.io import write_jsonl


def test_training_smoke_completes_on_tiny_tensor_bundle(tmp_path) -> None:
    feature_path = tmp_path / "features.npz"
    rng = np.random.default_rng(7)
    np.savez_compressed(
        feature_path,
        x_pre=rng.normal(size=(8, 5, 8)).astype("float32"),
        event_embedding=rng.normal(size=(8, 16)).astype("float32"),
        metadata=rng.normal(size=(8, 6)).astype("float32"),
        y=rng.normal(size=(8, 3)).astype("float32"),
        is_event=np.asarray([1, 1, 1, 1, 0, 0, 0, 0], dtype="float32"),
        split=np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype="int64"),
        event_ids=np.asarray([f"sample:{idx}" for idx in range(8)], dtype=object),
    )
    config = {
        "seed": 7,
        "model": {
            "price_features": 8,
            "metadata_features": 6,
            "event_embedding_dim": 16,
            "hidden_dim": 32,
            "latent_dim": 4,
            "output_dim": 3,
            "dropout": 0.0,
        },
        "loss": {},
        "training": {"epochs": 1, "batch_size": 4, "learning_rate": 0.001, "amp": False},
    }
    summary = train_model(config, feature_path, tmp_path, model_name="cebt")
    assert summary["epochs"] == 1
    assert (tmp_path / "cebt.pt").exists()
    metadata_path = tmp_path / "features_metadata.jsonl"
    write_jsonl(
        metadata_path,
        [
            {
                "sample_id": f"sample:{idx}",
                "feature_max_date": "2024-01-02",
                "label_start_date": "2024-01-03",
                "label_source": "future_returns_only",
            }
            for idx in range(8)
        ],
    )
    metrics = evaluate_model(
        config,
        feature_path,
        metadata_path,
        tmp_path / "cebt.pt",
        tmp_path,
        split=1,
    )
    assert metrics["rows"] == 4
    zero_metrics = evaluate_model(
        config,
        feature_path,
        metadata_path,
        tmp_path / "cebt.pt",
        tmp_path,
        split=1,
        intervention="zero_event",
    )
    assert zero_metrics["rows"] == 4
    assert (tmp_path / "cebt_zero_event_eval_metrics.json").exists()
