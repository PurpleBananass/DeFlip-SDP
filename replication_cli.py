"""Unified command-line entry points for the DeFlip experiments."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Tuple

BASE_DIR = Path(__file__).resolve().parent


@contextmanager
def study_context(study_folder: str):
    """Temporarily add a study folder to ``sys.path`` and yield its Path."""

    study_path = BASE_DIR / study_folder
    sys.path.insert(0, str(study_path))
    try:
        yield study_path
    finally:
        sys.path.remove(str(study_path))


def load_module(module_path: Path, name: str):
    """Load a module from ``module_path`` under a temporary name."""

    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_projects(projects: dict, requested: str) -> Iterable[str]:
    if requested == "all":
        return list(sorted(projects.keys()))
    return [project.strip() for project in requested.split(",") if project.strip()]


# ---------------------------- JIT-SDP tasks ---------------------------------

def jit_train_models(evaluate_only: bool = False):
    with study_context("JIT-SDP") as study_path:
        trainer = load_module(study_path / "train_models_jit.py", "jit_train")
        if evaluate_only:
            trainer.eval_all_project()
        else:
            trainer.train_all_project()


def jit_run_explanations(model_type: str, explainer_type: str, project: str):
    with study_context("JIT-SDP") as study_path:
        data_utils = load_module(study_path / "data_utils.py", "jit_data_utils")
        runner = load_module(study_path / "run_explainer.py", "jit_run_explainer")
        projects = data_utils.read_dataset()
        for project_name in resolve_projects(projects, project):
            train, test = projects[project_name]
            runner.run_single_project(train, test, project_name, model_type, explainer_type)


# ------------------------------ SDP tasks -----------------------------------

def sdp_preprocess():
    with study_context("SDP") as study_path:
        preprocess = load_module(study_path / "preprocess.py", "sdp_preprocess")
        preprocess.organize_original_dataset()
        preprocess.prepare_release_dataset()


def sdp_train_models(evaluate_only: bool = False):
    with study_context("SDP") as study_path:
        trainer = load_module(study_path / "train_models.py", "sdp_train")
        if evaluate_only:
            trainer.eval_all_project()
        else:
            trainer.train_all_project()


def sdp_run_explanations(model_type: str, explainer_type: str, project: str):
    with study_context("SDP") as study_path:
        data_utils = load_module(study_path / "data_utils.py", "sdp_data_utils")
        runner = load_module(study_path / "run_explainer.py", "sdp_run_explainer")
        projects = data_utils.read_dataset()
        for project_name in resolve_projects(projects, project):
            train, test = projects[project_name]
            runner.run_single_project(train, test, project_name, model_type, explainer_type)


# ----------------------------- CLI parsing ----------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="study", required=True)

    # JIT-SDP
    jit_parser = subparsers.add_parser("jit", help="JIT-SDP tasks")
    jit_subparsers = jit_parser.add_subparsers(dest="command", required=True)

    jit_train = jit_subparsers.add_parser("train-models", help="Train or evaluate JIT models")
    jit_train.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Skip training and only run evaluation on saved models.",
    )

    jit_explain = jit_subparsers.add_parser("explain", help="Run LIME/LIME-HPO explainers")
    jit_explain.add_argument(
        "--model-type",
        default="RandomForest",
        choices=["RandomForest", "SVM", "XGBoost", "LightGBM", "CatBoost"],
        help="Classifier to load for explanation.",
    )
    jit_explain.add_argument(
        "--explainer-type",
        default="LIME-HPO",
        choices=["LIME", "LIME-HPO"],
        help="Which explainer to run.",
    )
    jit_explain.add_argument(
        "--project",
        default="all",
        help="Project name or comma-separated list; use 'all' for every project.",
    )

    # SDP
    sdp_parser = subparsers.add_parser("sdp", help="Static defect prediction tasks")
    sdp_subparsers = sdp_parser.add_subparsers(dest="command", required=True)

    sdp_subparsers.add_parser("preprocess", help="Prepare release datasets")

    sdp_train = sdp_subparsers.add_parser("train-models", help="Train or evaluate SDP models")
    sdp_train.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Skip training and only run evaluation on saved models.",
    )

    sdp_explain = sdp_subparsers.add_parser(
        "explain", help="Run SDP explainers (LIME, LIME-HPO, TimeLIME, SQAPlanner)"
    )
    sdp_explain.add_argument(
        "--model-type",
        default="RandomForest",
        choices=["RandomForest", "XGBoost", "SVM"],
        help="Classifier to load for explanation.",
    )
    sdp_explain.add_argument(
        "--explainer-type",
        default="LIME-HPO",
        choices=["LIME", "LIME-HPO", "TimeLIME", "SQAPlanner"],
        help="Which explainer to run.",
    )
    sdp_explain.add_argument(
        "--project",
        default="all",
        help="Project name or comma-separated list; use 'all' for every project.",
    )

    return parser


def main(argv: Tuple[str, ...] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.study == "jit":
        if args.command == "train-models":
            jit_train_models(evaluate_only=args.evaluate_only)
        elif args.command == "explain":
            jit_run_explanations(args.model_type, args.explainer_type, args.project)

    elif args.study == "sdp":
        if args.command == "preprocess":
            sdp_preprocess()
        elif args.command == "train-models":
            sdp_train_models(evaluate_only=args.evaluate_only)
        elif args.command == "explain":
            sdp_run_explanations(args.model_type, args.explainer_type, args.project)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
