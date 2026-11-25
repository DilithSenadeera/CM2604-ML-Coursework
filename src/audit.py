"""Audit the fixed nominated model's test errors by demographic group."""

import json
from hashlib import sha256

import numpy as np
from sklearn.metrics import confusion_matrix

from churn_data import DATA_FILE, PROJECT_ROOT, load_data, split_data


def group_errors(labels, predicted, groups):
    """Return counts and error rates, keeping undefined rates as None."""
    labels = np.asarray(labels)
    predicted = np.asarray(predicted)
    groups = np.asarray(groups)
    results = {}
    for group in np.unique(groups):
        members = groups == group
        tn, fp, fn, tp = confusion_matrix(labels[members], predicted[members], labels=[0, 1]).ravel()
        positives = int(tp + fn)
        negatives = int(tn + fp)
        results[str(group)] = {
            "customers": int(members.sum()),
            "churned": positives,
            "stayed": negatives,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "recall": float(tp / positives) if positives else None,
            "false_positive_rate": float(fp / negatives) if negatives else None,
        }
    return results


def main():
    report_file = PROJECT_ROOT / "results" / "test.json"
    report_bytes = report_file.read_bytes()
    report = json.loads(report_bytes)
    if sha256(DATA_FILE.read_bytes()).hexdigest() != report["data_sha256"]:
        raise ValueError("The source dataset differs from the evaluated dataset.")
    _, test = split_data(load_data()[0])
    nominee = report["nominated_before_test"]
    threshold = report["threshold"]
    with np.load(PROJECT_ROOT / "data" / "test_predictions.npz") as saved:
        np.testing.assert_array_equal(saved["customer_id"], test["customerID"])
        np.testing.assert_array_equal(saved["labels"], test["Churn"])
        np.testing.assert_array_equal(saved["gender"], test["gender"])
        np.testing.assert_array_equal(saved["senior_citizen"], test["SeniorCitizen"])
        model_index = list(saved["model_names"]).index(nominee)
        predicted = (saved["probabilities"][model_index] >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(test["Churn"], predicted, labels=[0, 1]).ravel()
    np.testing.assert_array_equal(
        [tn, fp, fn, tp], [report["models"][nominee][key] for key in ["tn", "fp", "fn", "tp"]]
    )
    results = {
        "model": nominee,
        "threshold": threshold,
        "test_report_sha256": sha256(report_bytes).hexdigest(),
        "groups": {},
    }
    for field in ["gender", "SeniorCitizen"]:
        results["groups"][field] = group_errors(test["Churn"], predicted, test[field])
        for group, counts in results["groups"][field].items():
            print(field, group, counts)

    output_file = PROJECT_ROOT / "results" / "subgroups.json"
    output_file.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
