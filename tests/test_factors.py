from __future__ import annotations

from cebt.data.factors import FactorRow, load_factor_rows


def test_load_factor_rows_from_jsonl_cache(tmp_path) -> None:
    path = tmp_path / "factors.jsonl"
    path.write_text(
        "\n".join(
            [
                (
                    '{"date":"2024-01-02","mkt_rf":0.001,"smb":0.0002,'
                    '"hml":-0.0001,"rf":0.00005,"mom":0.0003,"source_url":"unit-test"}'
                ),
                (
                    '{"date":"2024-01-03","mkt_rf":-0.002,"smb":0.0001,'
                    '"hml":0.0004,"rf":0.00005,"mom":null,"source_url":"unit-test"}'
                ),
            ]
        ),
        encoding="utf-8",
    )

    rows = load_factor_rows(path)

    assert len(rows) == 2
    assert isinstance(rows[0], FactorRow)
    assert rows[0].mkt_rf == 0.001
    assert rows[0].mom == 0.0003
    assert rows[1].mom is None
