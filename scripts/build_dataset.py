"""Build supervised MTFF-Net arrays from a private aligned table."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from mtff_net.config import load_config
from mtff_net.data import build_supervised_dataset, read_database_table, save_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-csv", type=Path, help="Private aligned monitoring CSV.")
    source.add_argument("--database-url", help="SQLAlchemy database URL for a private database.")
    parser.add_argument("--query", help="SQL query used with --database-url.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dataset"))
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--window", type=int)
    parser.add_argument("--horizon", type=int)
    args = parser.parse_args()

    cfg = load_config(args.config).dataset
    window = args.window or cfg.window
    horizon = args.horizon or cfg.horizon

    if args.input_csv:
        df = pd.read_csv(args.input_csv)
    else:
        if not args.query:
            raise SystemExit("--query is required when --database-url is used.")
        df = read_database_table(args.database_url, args.query)

    dataset = build_supervised_dataset(
        df,
        window=window,
        horizon=horizon,
        datetime_column=cfg.datetime_column,
        target_prefixes=cfg.target_prefixes,
        feature_prefixes=cfg.feature_prefixes,
        extra_feature_columns=cfg.extra_feature_columns,
    )
    save_dataset(dataset, args.output_dir)

    metadata = {
        "dataset": asdict(cfg),
        "window": dataset.window,
        "horizon": dataset.horizon,
        "n_sequences": int(dataset.X.shape[0]),
        "n_features": int(dataset.X.shape[-1]),
        "n_targets": int(dataset.y.shape[-1]),
        "feature_columns": dataset.feature_columns,
        "target_columns": dataset.target_columns,
        "target_indices": dataset.target_indices,
        "start_time": str(dataset.times.min()),
        "end_time": str(dataset.times.max()),
        "release_note": "Generated from a private measured monitoring table. Do not commit generated arrays.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({k: metadata[k] for k in ["n_sequences", "n_features", "n_targets", "start_time", "end_time"]}, indent=2))


if __name__ == "__main__":
    main()

