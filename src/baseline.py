"""Compare untuned models using cross-validation on the development sample."""

import json
import platform
import warnings
from hashlib import sha256
from importlib.metadata import version

import numpy as np
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from churn_data import DATA_FILE, PROJECT_ROOT, SEED, load_data, split_data
from models import FEATURE_COLUMNS, THRESHOLD, baseline_models, score_probabilities


RESULT_FILE = PROJECT_ROOT / "results" / "baseline.json"
METRICS = ["average_precision", "roc_auc", "precision", "recall", "f1", "accuracy"]


def main():
    customers, _ = load_data()
    development, reserved = split_data(customers)
    features = development[FEATURE_COLUMNS]
    labels = development["Churn"]
    cross_validation = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(cross_validation.split(features, labels))
    results = {
        "data_sha256": sha256(DATA_FILE.read_bytes()).hexdigest(),
        "python_version": platform.python_version(),
        "packages": {name: version(name) for name in ["numpy", "pandas", "scikit-learn", "scipy"]},
        "seed": SEED,
        "development_rows": len(development),
        "reserved_rows": len(reserved),
        "folds": len(folds),
        "threshold": THRESHOLD,
        "feature_columns": FEATURE_COLUMNS,
        "models": {},
    }

    for name, pipeline in baseline_models().items():
        fold_results = []
        for fold_number, (fit_rows, validation_rows) in enumerate(folds, start=1):
            train_features = features.iloc[fit_rows]
            train_labels = labels.iloc[fit_rows]
            validation_features = features.iloc[validation_rows]
            validation_labels = labels.iloc[validation_rows]

            # Start with an unfitted pipeline so each fold learns its own preprocessing.
            fitted = clone(pipeline)
            # A failed optimisation must be fixed before accepting any scores.
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                fitted.fit(train_features, train_labels)

            # Column 1 is the probability of churn because labels are encoded as 0 and 1.
            probabilities = fitted.predict_proba(validation_features)[:, 1]
            scores = score_probabilities(validation_labels, probabilities)
            fit_probabilities = fitted.predict_proba(train_features)[:, 1]
            scores["fold"] = fold_number
            scores["fit_rows"] = len(train_features)
            scores["validation_rows"] = len(validation_features)
            scores["validation_churned"] = int(validation_labels.sum())
            scores["fit_average_precision"] = float(
                average_precision_score(train_labels, fit_probabilities)
            )
            fold_results.append(scores)
            print(f"{name}, fold {fold_number}: validation AP = {scores['average_precision']:.4f}",
                  flush=True)

        summary = {}
        for metric in METRICS + ["fit_average_precision"]:
            values = [fold[metric] for fold in fold_results]
            summary[metric] = {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1))}

        confusion_totals = {}
        for count in ["tn", "fp", "fn", "tp"]:
            confusion_totals[count] = sum(fold[count] for fold in fold_results)

        results["models"][name] = {
            "classifier_parameters": pipeline.named_steps["classifier"].get_params(),
            "fold_scores": fold_results,
            "summary": summary,
            "validation_confusion_totals": confusion_totals,
        }

    RESULT_FILE.parent.mkdir(exist_ok=True)
    RESULT_FILE.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved {RESULT_FILE.name}. Metrics are validation fold means with sample standard deviations.")
    for name, result in results["models"].items():
        ap = result["summary"]["average_precision"]
        print(f"{name}: AP = {ap['mean']:.4f} +/- {ap['sd']:.4f}")


if __name__ == "__main__":
    main()
