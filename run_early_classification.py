"""Run focused, non-mutating early-prediction model comparisons."""

import json
import argparse
from pathlib import Path

import pandas as pd

from src.early_classification import run_early_classification, save_early_classification_report
from src.validation import profile_dataset

ROOT = Path(__file__).resolve().parent
DATASETS = ("student-mat.csv", "student-por.csv")
EARLY_BASELINES = {
    "student-mat.csv": "early_classification_20261028-170425-466624.json",
    "student-por.csv": "early_classification_20261028-173530-966101.json",
}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--dataset",
    choices=DATASETS,
    action="append",
    help="Run only a selected dataset; may be supplied more than once.",
)
args = parser.parse_args()
datasets = tuple(args.dataset) if args.dataset else DATASETS


def historical_baselines(dataset: str) -> dict:
    """Capture prior early-classification and Phase 6 metrics before new reports exist."""
    early_path = ROOT / "reports" / EARLY_BASELINES[dataset]
    early_report = json.loads(early_path.read_text(encoding="utf-8"))

    phase6_reports = sorted(
        (ROOT / "reports").glob("phase6_validation_*.json"),
        key=lambda path: path.stat().st_mtime,
    )
    early_experiments = []
    phase6_source = None
    if phase6_reports:
        phase6_source = phase6_reports[-1]
        phase6 = json.loads(phase6_source.read_text(encoding="utf-8"))
        early_experiments = [
            experiment
            for experiment in phase6.get("experiments", [])
            if experiment.get("dataset") == dataset
            and experiment.get("task") == "Classification"
            and experiment.get("scenario") == "early"
        ]

    result = {
        "early_classification": {
            "source": early_path.name,
            "summary": early_report.get("summary", []),
        },
        "phase6": {
            "source": phase6_source.name if phase6_source else None,
            "experiments": [
                {
                    "selected_model": max(
                        experiment.get("cv_summary", []),
                        key=lambda row: row.get("selected_seed_count", 0),
                    ).get("model"),
                    "held_out_summary": experiment.get("held_out_summary", []),
                }
                for experiment in early_experiments
            ],
        },
    }
    return result


references = {name: historical_baselines(name) for name in datasets}

for name in datasets:
    dataset_path = ROOT / "data" / "raw" / name
    profile_dataset(dataset_path)
    frame = pd.read_csv(dataset_path, sep=";")
    report = run_early_classification(
        frame,
        dataset=name,
        seeds=(42, 43, 44),
        test_size=0.20,
        folds=5,
        optuna_trials=8,
    )
    report["historical_baselines"] = references[name]
    path = save_early_classification_report(report, ROOT / "reports")
    print(f"{name}: {path} and {path.with_suffix('.csv')}")
    best = report["best_model_by_mean_training_cv_macro_f1"]
    print(
        f"  CV-selected: {best['model']} "
        f"(training CV macro-F1={best['training_cv_macro_f1_mean']:.4f} +/- "
        f"{best['training_cv_macro_f1_std_across_seeds']:.4f}; "
        f"held-out accuracy={best['held_out_accuracy_mean']:.4f} +/- "
        f"{best['held_out_accuracy_std']:.4f}; "
        f"held-out macro-F1={best['held_out_macro_f1_mean']:.4f} +/- "
        f"{best['held_out_macro_f1_std']:.4f})"
    )
    print(f"  Per-seed CV winners: {report['selected_model_by_seed']}")
