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

- Derive actionable plans from previously saved explanations (optionally include a
  search strategy suffix if you used one during explanation generation):

  ```bash
  python replication_cli.py jit plan-actions \
      --model-type RandomForest \
      --explainer-type LIME-HPO \
      --project all \
      --search-strategy confidence
  ```

- Apply plans to flip predictions (run this before plotting):

  ```bash
  python replication_cli.py jit flip \
      --model-type RandomForest \
      --explainer-type LIME-HPO \
      --project all \
      --search-strategy confidence
  ```

- Summarize flip experiments for RQ tables/plots (defaults to running all RQs; add
  `--closest` to use the closest-plan pipeline and `--use-default-groups` for RQ3):

  ```bash
  python replication_cli.py jit evaluate --explainer all --models all --projects all
  ```

### SDP

For the SDP study, some explainers require extra steps (e.g., rule mining for
SQAPlanner) and DeFlip counterfactuals are generated via `niceml.py`. The CLI
captures each stage explicitly:

- Preprocess the raw dataset into per-release splits (requires R):

  ```bash
  python replication_cli.py sdp preprocess
  ```

- Train all models (or re-evaluate saved models with `--evaluate-only`):

  ```bash
  python replication_cli.py sdp train-models
  ```

- Run LIME/LIME-HPO/TimeLIME/SQAPlanner explanations:

  ```bash
  python replication_cli.py sdp explain \
      --model-type RandomForest \
      --explainer-type LIME-HPO \
      --project all
  ```

- (SQAPlanner only) Mine association rules on the generated instances before
  creating plans. BigML credentials must be present in `.env`:

  ```bash
  python replication_cli.py sdp mine-rules \
      --model-type RandomForest \
      --search-strategy confidence \
      --project activemq@2
  ```

- Convert explanation outputs into closest plans (works for LIME, LIME-HPO,
  TimeLIME, and SQAPlanner; include `--search-strategy` if your explanations
  were stored under a subfolder such as `confidence/`):

  ```bash
  python replication_cli.py sdp plan-actions \
      --model-type RandomForest \
      --explainer-type LIME-HPO \
      --project all \
      --search-strategy confidence
  ```

  Use `--compute-importance` to compute feature-importance ratios instead of
  writing plan files.

- Generate DeFlip/NICE counterfactuals (writes `experiments/{project}/{model}/CF_all.csv`):

  ```bash
  python replication_cli.py sdp counterfactuals \
      --project all \
      --model-types RandomForest,SVM,XGBoost \
      --max-features 5 \
      --distance unit_l2
  ```

- Apply plans to flip predictions (run this before plotting the plan-based
  explainers). Add `--closest` to read from `plans_closest/` and
  `experiments_closest/`:

  ```bash
  python replication_cli.py sdp flip \
      --model-type RandomForest \
      --explainer-type LIME-HPO \
      --project all \
      --search-strategy confidence
  ```

- Summarize flip experiments and DeFlip counterfactuals for plotting:

  ```bash
  python replication_cli.py sdp evaluate --explainer all --models all --projects all
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
- The overall workflow for actionability is: (1) run explainers, (2) generate plans using
  the new `plan-actions` commands, and (3) evaluate/flip using the study-specific scripts
  such as `SDP/evaluate_cf.py` or the `evaluate_*` utilities under `JIT-SDP/`. Some
  explainers require extra manual steps (e.g., SQAPlanner needs BigML credentials for
  rule mining via `SDP/mining_sqa_rules.py`). Refer to the inline comments inside each
  script for the exact sequencing.
