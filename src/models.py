"""Baseline classifiers, preprocessing and common evaluation metrics."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_data import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, SEED


FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
THRESHOLD = 0.50


def make_pipeline(classifier, scale_numeric):
    """Learn numeric scaling and categorical encoding together with a classifier."""
    numeric_transform = StandardScaler() if scale_numeric else "passthrough"
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    preprocessing = ColumnTransformer(
        [
            ("numeric", numeric_transform, NUMERIC_COLUMNS),
            ("categorical", encoder, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocessing", preprocessing), ("classifier", classifier)])


def baseline_models():
    """Create the predefined baselines with explicit settings."""
    return {
        "Dummy": make_pipeline(DummyClassifier(strategy="prior"), scale_numeric=False),
        "KNN": make_pipeline(
            KNeighborsClassifier(n_neighbors=5, weights="uniform", metric="euclidean"),
            scale_numeric=True,
        ),
        "Logistic regression": make_pipeline(
            LogisticRegression(C=1.0, l1_ratio=0.0, solver="lbfgs", max_iter=1000),
            scale_numeric=True,
        ),
        "Random forest": make_pipeline(
            RandomForestClassifier(
                n_estimators=100, max_depth=None, min_samples_leaf=1,
                max_features="sqrt", random_state=SEED, n_jobs=1,
            ),
            scale_numeric=False,
        ),
    }


def score_probabilities(labels, probabilities):
    """Score churn probabilities with the same positive class and threshold."""
    probabilities = np.asarray(probabilities)
    outside_range = (probabilities < 0) | (probabilities > 1)
    if not np.isfinite(probabilities).all() or outside_range.any():
        raise ValueError("Churn probabilities must be finite values between zero and one.")
    predicted = (probabilities >= THRESHOLD).astype(int)
    # Rows are actual labels and columns are predictions, ordered as stayed (0), churned (1).
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {
        "average_precision": float(average_precision_score(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "accuracy": float(accuracy_score(labels, predicted)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
