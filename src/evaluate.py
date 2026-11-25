"""Evaluate the fixed baseline and tuned settings on the reserved test sample."""

import json
import platform
import warnings
from hashlib import sha256
from importlib.metadata import version

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

from churn_data import DATA_FILE, PROJECT_ROOT, SEED, load_data, split_data
from models import FEATURE_COLUMNS, THRESHOLD, baseline_models, score_probabilities
from tune import TUNING_FILE


def make_figure(labels, probabilities, nominee):
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.6), layout="constrained")
    for name, scores in probabilities.items():
        if name.endswith("tuned"):
            RocCurveDisplay.from_predictions(labels, scores, name=name, ax=axes[0])
            PrecisionRecallDisplay.from_predictions(labels, scores, name=name, ax=axes[1])
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="grey")
    axes[1].axhline(labels.mean(), linestyle="--", color="grey", label="Churn proportion")
    predicted = (probabilities[nominee] >= THRESHOLD).astype(int)
    ConfusionMatrixDisplay.from_predictions(
        labels, predicted, labels=[0, 1], display_labels=["Stayed", "Churned"],
        ax=axes[2], colorbar=False, cmap="Blues",
    )
    axes[0].set_title("Test ROC curves")
    axes[1].set_title("Test precision and recall")
    axes[2].set_title(f"{nominee}\nThreshold = {THRESHOLD:.2f}")
    for axis in axes[:2]:
        axis.legend(fontsize=8)
    figure.suptitle(f"Held-out customer evaluation (n = {len(labels):,})", fontsize=15)
    figure.savefig(PROJECT_ROOT / "results" / "test-evaluation.png", dpi=180)
    plt.close(figure)


def main():
    tuning_bytes = TUNING_FILE.read_bytes()
    tuning = json.loads(tuning_bytes)
    if sha256(DATA_FILE.read_bytes()).hexdigest() != tuning["data_sha256"]:
        raise ValueError("Dataset differs from the tuning experiment.")
    if tuning["seed"] != SEED or tuning["threshold"] != THRESHOLD or tuning["feature_columns"] != FEATURE_COLUMNS:
        raise ValueError("Split, threshold or predictors differ from the tuning experiment.")

    development, test = split_data(load_data()[0])
    if set(development["customerID"]) & set(test["customerID"]):
        raise ValueError("A customer appears in both development and test samples.")
    train_features = development[FEATURE_COLUMNS]
    train_labels = development["Churn"]
    test_features = test[FEATURE_COLUMNS]
    test_labels = test["Churn"]
    nominee = tuning["nominated_model"] + " tuned"
    results = {
        "data_sha256": tuning["data_sha256"],
        "tuning_sha256": sha256(tuning_bytes).hexdigest(),
        "python_version": platform.python_version(),
        "packages": {name: version(name) for name in ["numpy", "pandas", "scikit-learn", "scipy"]},
        "seed": SEED,
        "threshold": THRESHOLD,
        "development_rows": len(development),
        "test_rows": len(test),
        "test_churned": int(test_labels.sum()),
        "nominated_before_test": nominee,
        "models": {},
    }
    probabilities = {}
    for name, baseline in baseline_models().items():
        settings = {"baseline": baseline}
        if name != "Dummy":
            parameters = tuning["models"][name]["selected"]["parameters"]
            settings["tuned"] = clone(baseline).set_params(**parameters)
        for setting, pipeline in settings.items():
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                pipeline.fit(train_features, train_labels)
            label = f"{name} {setting}"
            probabilities[label] = pipeline.predict_proba(test_features)[:, 1]
            scores = score_probabilities(test_labels, probabilities[label])
            results["models"][label] = scores
            print(f"{label}: test AP = {scores['average_precision']:.4f}, F1 = {scores['f1']:.4f}", flush=True)

    result_file = PROJECT_ROOT / "results" / "test.json"
    result_file.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    # Keep customer-level predictions local for verification and the later subgroup audit.
    np.savez(
        PROJECT_ROOT / "data" / "test_predictions.npz",
        customer_id=test["customerID"].to_numpy(dtype=str),
        labels=test_labels.to_numpy(),
        gender=test["gender"].to_numpy(dtype=str),
        senior_citizen=test["SeniorCitizen"].to_numpy(dtype=str),
        model_names=np.array(list(probabilities)),
        probabilities=np.array(list(probabilities.values())),
    )
    make_figure(test_labels, probabilities, nominee)


if __name__ == "__main__":
    main()
