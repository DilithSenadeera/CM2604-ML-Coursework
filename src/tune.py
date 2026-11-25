"""Select hyperparameters using only the development sample."""

import json
import warnings
from hashlib import sha256

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from churn_data import DATA_FILE, PROJECT_ROOT, SEED, load_data, split_data
from models import FEATURE_COLUMNS, THRESHOLD, baseline_models


PARAMETER_GRIDS = {
    "KNN": {
        "classifier__n_neighbors": [5, 11, 21, 31],
        "classifier__weights": ["uniform", "distance"],
        "classifier__metric": ["euclidean", "manhattan"],
    },
    "Logistic regression": {"classifier__C": [1.0, 0.01, 0.1, 10.0, 100.0]},
    "Random forest": {
        "classifier__n_estimators": [100, 300],
        "classifier__max_depth": [None, 6, 12],
        "classifier__min_samples_leaf": [1, 5, 15],
    },
}
TUNING_FILE = PROJECT_ROOT / "results" / "tuning.json"


def main():
    baseline = json.loads((PROJECT_ROOT / "results" / "baseline.json").read_text())
    data_hash = sha256(DATA_FILE.read_bytes()).hexdigest()
    if data_hash != baseline["data_sha256"]:
        raise ValueError("Dataset differs from the baseline experiment.")

    development, reserved = split_data(load_data()[0])
    features = development[FEATURE_COLUMNS]
    labels = development["Churn"]
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(features, labels))
    results = {
        "data_sha256": data_hash,
        "seed": SEED,
        "threshold": THRESHOLD,
        "feature_columns": FEATURE_COLUMNS,
        "development_rows": len(development),
        "reserved_rows": len(reserved),
        "models": {},
    }

    for name, pipeline in baseline_models().items():
        if name == "Dummy":
            continue
        print(f"Searching {name}...", flush=True)
        search = GridSearchCV(
            pipeline, PARAMETER_GRIDS[name], scoring="average_precision", cv=folds,
            refit=False, return_train_score=True, error_score="raise", n_jobs=1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            search.fit(features, labels)

        candidates = []
        for index, parameters in enumerate(search.cv_results_["params"]):
            validation_ap = []
            fit_ap = []
            for fold in range(len(folds)):
                validation_ap.append(float(search.cv_results_[f"split{fold}_test_score"][index]))
                fit_ap.append(float(search.cv_results_[f"split{fold}_train_score"][index]))
            candidates.append({
                "parameters": parameters,
                "validation_ap": validation_ap,
                "fit_ap": fit_ap,
                "mean_validation_ap": float(np.mean(validation_ap)),
                "sd_validation_ap": float(np.std(validation_ap, ddof=1)),
            })

        # The first candidate is the baseline, so an exact tie keeps the baseline.
        baseline_ap = [fold["average_precision"] for fold in baseline["models"][name]["fold_scores"]]
        np.testing.assert_allclose(candidates[0]["validation_ap"], baseline_ap, rtol=0, atol=1e-12)
        selected_index = int(np.argmax([candidate["mean_validation_ap"] for candidate in candidates]))
        results["models"][name] = {
            "selected_index": selected_index,
            "selected": candidates[selected_index],
            "candidates": candidates,
        }
        print(f"Selected {candidates[selected_index]['parameters']}", flush=True)

    results["nominated_model"] = max(
        results["models"], key=lambda name: results["models"][name]["selected"]["mean_validation_ap"]
    )
    TUNING_FILE.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved tuning choices before test evaluation. Nominee: {results['nominated_model']}")


if __name__ == "__main__":
    main()
