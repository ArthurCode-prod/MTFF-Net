# MTFF-Net

MTFF-Net is an open-source implementation scaffold for joint crack-width and
rebar-stress forecasting from structural health monitoring time series. The
model uses a shared temporal encoder with task-specific residual heads so that
crack and steel-stress responses are learned together while preserving separate
output channels for each sensor group.

This repository accompanies the manuscript submitted to *Buildings*. The real
monitoring dataset used in the study is not included because it is subject to
project confidentiality and data-owner restrictions. The small toy data under
`data/sample/` are provided only to validate the code path and are not used for
paper experiments.

## Repository Layout

```text
MTFF-Net/
  configs/default.yaml          # Reference training and dataset settings
  data/README.md                # Required data schema and release policy
  data/sample/                  # Tiny toy schema example, not research data
  examples/quickstart.py        # In-memory smoke test
  scripts/build_dataset.py      # CSV or database table to supervised arrays
  scripts/train.py              # Train MTFF-Net on prepared arrays
  scripts/evaluate.py           # Evaluate a saved checkpoint
  src/mtff_net/                 # Reusable Python package
```

## Installation

```bash
git clone https://github.com/ArthurCode-prod/MTFF-Net.git
cd MTFF-Net
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev,db]"
```

Linux/macOS users can activate the environment with
`source .venv/bin/activate`.

## Quick Smoke Test

Run the toy example to confirm that the package, dataloader, model, and training
loop are wired correctly:

```bash
python examples/quickstart.py
```

The script creates a tiny synthetic dataframe in memory. It does not reproduce
or approximate the manuscript dataset.

## Preparing a Private Dataset

MTFF-Net expects an aligned monitoring table with one timestamp column and
numeric feature/target columns. The default naming convention is:

| Column pattern | Meaning |
| --- | --- |
| `datetime` | Response timestamp after conservative alignment |
| `crack_*` | Crack-width response targets |
| `rebar_*` | Rebar-stress response targets |
| `temperature_*` | Sensor temperature or local thermal variables |
| `water_level`, `rainfall_*`, `environment_*` | External hydraulic or environmental drivers |

Build supervised arrays from a private CSV:

```bash
python scripts/build_dataset.py \
  --input-csv path/to/private_aligned_monitoring.csv \
  --output-dir outputs/private_dataset \
  --window 21 \
  --horizon 6
```

Build from a database query:

```bash
python scripts/build_dataset.py \
  --database-url "postgresql+psycopg2://USER:PASS@HOST:5432/DB" \
  --query "select * from aligned_monitoring_view order by datetime" \
  --output-dir outputs/private_dataset
```

Do not commit private datasets, generated arrays, checkpoints, or credentials.
The `.gitignore` file excludes the common generated artifacts by default.

## Training

```bash
python scripts/train.py \
  --dataset outputs/private_dataset/dataset_arrays.npz \
  --metadata outputs/private_dataset/dataset_metadata.json \
  --output-dir outputs/experiment \
  --config configs/default.yaml
```

The training script writes a checkpoint, per-group metrics, and prediction CSVs
to the selected output directory. Reported manuscript results should always be
computed from the private measured dataset, not from the toy sample.

## Citation

If you use this code, please cite the associated manuscript when it becomes
available and include the repository URL:

```bibtex
@software{mtff_net_2026,
  title  = {MTFF-Net: Multi-Task Temporal Fusion for Crack and Rebar-Stress Forecasting},
  author = {Liu, Binbin and Wang, Mingming and Zhu, Xiaolei and Zhang, Wanbo},
  year   = {2026},
  url    = {https://github.com/ArthurCode-prod/MTFF-Net}
}
```

