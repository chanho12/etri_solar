# SOLAR End-to-End Code

This is the supplied raw-to-submission SOLAR implementation. It constructs personalized features, trains the long-term anchor, learns day-specific residual corrections, and applies target-specific refinement.

The Python code and pinned dependencies are preserved from `submit_final_end2end`. No raw participant data or precomputed outputs are bundled. The accompanying [model description](../docs/MODEL_DESCRIPTION.md), [paper](https://drive.google.com/file/d/18jN1ZuiLQeukAqOCAjvDY2MZCu7U_5IE/view?usp=sharing), and [model report](../docs/model-report.pdf) describe the supplied method and results.

## Requirements

- 64-bit Linux and a POSIX shell
- Miniconda or Anaconda
- Python 3.12 (created by `setup_conda.sh`)
- Official challenge CSVs and 12 raw sensor Parquet files; see [data layout](data/README.md)

The recorded development environment used Ubuntu 22.04.3 LTS and an NVIDIA RTX A6000 (48 GB). CPU execution may take substantially longer. The website can be viewed on any modern browser; model training has not been re-run as part of this website update.

## Setup and run

From this `code/` directory:

```bash
chmod +x setup_conda.sh run_solar.sh
./setup_conda.sh
./run_solar.sh
```

`setup_conda.sh` creates a project-local `.conda-env` and installs the exact versions in `requirements.txt`. It checks imports and dependency consistency. `run_solar.sh` runs that environment without requiring shell activation.

The alternative, after activating the environment, is:

```bash
conda activate "$PWD/.conda-env"
python run.py --no-frozen
```

The final output is `data/submissions/submission_final.csv`. The supplied pipeline checks input hashes, output column order, 250 rows, missing values, probability ranges, and the final SHA-256.

## Optional flags

```bash
# Resume artifacts created by the same code and environment
./run_solar.sh --resume --reuse-feature-cache

# Include the optional MIS-LSTM candidate
./run_solar.sh --include-mis-lstm
```

For the first run, use `./run_solar.sh` without reuse options. Historical caches may contain incompatible source names.

## Files

| File | Purpose |
|---|---|
| `run.py` | Deterministic execution entry point |
| `solar_pipeline.py` | Three-stage training orchestration |
| `solar_core.py` | Sensor aggregation, features, modeling, stacking, refinement |
| `setup_conda.sh` | Create and validate the environment |
| `run_solar.sh` | Run training in the isolated environment |
| `environment.yml` | Conda environment definition |
| `requirements.txt` | Pinned Python dependencies |
| `data/README.md` | Required input layout |

Generated features, model caches, submissions, and the Conda environment are ignored by Git.
