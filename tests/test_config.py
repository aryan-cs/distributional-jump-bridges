from __future__ import annotations

from cebt.utils.config import load_config


def test_config_extends_base_config_and_merges_model_config() -> None:
    config = load_config("configs/paper_v3_bge_ff4.yaml")

    assert config["project"]["run_name"] == "paper_v3_bge_ff4"
    assert config["data"]["forms"] == ["8-K"]
    assert config["features"]["embedding"]["model_id"] == "BAAI/bge-small-en-v1.5"
    assert config["labels"]["mode"] == "ff4"
    assert config["model"]["event_embedding_dim"] == 384
