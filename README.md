# DeFlip-SDP Replication Package

This repository contains the experiment code for the DeFlip study across two settings:

- **JIT-SDP** (`JIT-SDP/`): Just-in-time defect prediction.
- **SDP** (`SDP/`): Traditional static defect prediction.

The goal of this cleanup is to make it easier to reproduce the experiments, understand the
code layout, and orchestrate common tasks from a single command-line interface.

## Repository Layout

- `JIT-SDP/` – Training scripts, explainers, counterfactual generators, and utilities for
  the JIT study. Dataset and output locations are configured in `JIT-SDP/hyparams.py`.
- `SDP/` – Training, preprocessing, and explainer code for the SDP study. Paths live in
  `SDP/hyparams.py`.
- `plot_rq1.py`, `plot_rq2.py`, `plot_rq3.py` – Plotting utilities used by the paper.
- `replication_cli.py` – A new orchestration CLI that wraps the key pipelines for both
  studies.
- `requirements.txt` – Python dependencies for running the experiments.

## Environment Setup

1. Create and activate a Python 3.10+ environment.
2. Install the Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. (SDP preprocessing only) Install the R package [`Rnalytica`](https://cran.r-project.org/package=Rnalytica)
   and ensure `rpy2` can locate your R installation.

## Datasets

Dataset roots are configured in each study's `hyparams.py` file. Update the paths there to
point to your local copies before running the pipelines. Common locations include:

- **JIT-SDP:**
  - Raw CSV: `JIT-SDP/hyparams.py::JIT_DATASET_PATH`
  - Preprocessed per-release data: `JIT-SDP/hyparams.py::RELEASE_DATASET`
- **SDP:**
  - Original per-project CSVs: `SDP/hyparams.py::ORIGINAL_DATASET`
  - Preprocessed release splits: `SDP/hyparams.py::RELEASE_DATASET`

## Command-Line Orchestration

`replication_cli.py` exposes consistent entry points for both studies. Run `--help` on any
command to see available options.

### JIT-SDP

- Train all models:

  ```bash
  python replication_cli.py jit train-models
  ```

- Evaluate previously trained models:

  ```bash
  python replication_cli.py jit train-models --evaluate-only
  ```

- Run LIME/LIME-HPO explanations for a specific or all projects:

  ```bash
  python replication_cli.py jit explain \
      --model-type RandomForest \
      --explainer-type LIME-HPO \
      --project all
  ```

### SDP

- Preprocess the raw dataset into per-release splits (requires R):

  ```bash
  python replication_cli.py sdp preprocess
  ```

- Train all models:

  ```bash
  python replication_cli.py sdp train-models
  ```

- Evaluate previously trained models:

  ```bash
  python replication_cli.py sdp train-models --evaluate-only
  ```

- Run LIME/LIME-HPO/TimeLIME/SQAPlanner explanations:

  ```bash
  python replication_cli.py sdp explain \
      --model-type RandomForest \
      --explainer-type LIME-HPO \
      --project all
  ```

## Plotting

Once experiments finish, the `plot_rq*.py` scripts can be used to regenerate the figures
from the paper. Each script reads the CSV outputs written by the training/explanation
pipelines (see the constants in the script bodies for expected locations).

## Notes and Tips

- Paths in the study-specific `hyparams.py` files are relative to the repository root by
  default. Adjust them as needed for your environment.
- The explainers use multiprocessing; ensure your machine has enough memory to spawn
  concurrent workers when running the explanation commands.
- For quick smoke tests, you can point the dataset paths to a reduced sample and run the
  CLI with the same commands shown above.
