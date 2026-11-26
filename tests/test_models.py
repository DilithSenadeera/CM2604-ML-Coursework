"""Targeted checks for threshold consistency and preprocessing leakage."""

import unittest
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from churn_data import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS
from models import baseline_models, score_probabilities
from sklearn.model_selection import ParameterGrid
from tune import PARAMETER_GRIDS
from audit import group_errors


class BaselineChecks(unittest.TestCase):
    def test_subgroup_rates_and_undefined_denominators(self):
        counts = group_errors([0, 0, 1, 1], [0, 1, 0, 1], ["A"] * 4)["A"]
        self.assertEqual(counts["customers"], 4)
        self.assertEqual(counts["recall"], 0.5)
        self.assertEqual(counts["false_positive_rate"], 0.5)
        for key in ["tn", "fp", "fn", "tp"]:
            self.assertEqual(counts[key], 1)

        groups = group_errors([0, 0, 1, 1], [0, 1, 0, 1], ["A", "A", "B", "B"])
        self.assertIsNone(groups["A"]["recall"])
        self.assertIsNone(groups["B"]["false_positive_rate"])

    def test_search_grids_include_baseline_first(self):
        expected_sizes = {"KNN": 16, "Logistic regression": 5, "Random forest": 18}
        for name, grid in PARAMETER_GRIDS.items():
            with self.subTest(model=name):
                candidates = list(ParameterGrid(grid))
                self.assertEqual(len(candidates), expected_sizes[name])
                parameters = baseline_models()[name].get_params()
                for parameter, value in candidates[0].items():
                    self.assertEqual(value, parameters[parameter])

    def test_threshold_and_metrics_against_hand_calculation(self):
        # A score of exactly 0.50 must count as churn in every model.
        scores = score_probabilities([0, 0, 1, 1], [0.10, 0.50, 0.40, 0.90])
        self.assertEqual([scores[key] for key in ["tn", "fp", "fn", "tp"]], [1, 1, 1, 1])
        for metric in ["precision", "recall", "f1", "accuracy"]:
            self.assertEqual(scores[metric], 0.50)
        self.assertAlmostEqual(scores["roc_auc"], 0.75)
        self.assertAlmostEqual(scores["average_precision"], (1 + 2 / 3) / 2)

        dummy_scores = score_probabilities([0, 0, 1, 1], [0.25] * 4)
        self.assertEqual(dummy_scores["precision"], 0)
        self.assertEqual(dummy_scores["recall"], 0)

    def test_pipeline_learns_only_from_fitted_rows(self):
        # Fabricated rows make the training means and unseen category unambiguous.
        features = pd.DataFrame({column: ["Seen"] * 8 for column in CATEGORICAL_COLUMNS})
        features["tenure"] = [1, 2, 3, 4, 5, 6, 7, 8]
        features["MonthlyCharges"] = [10, 20, 30, 40, 50, 60, 70, 80]
        features["TotalCharges"] = features["tenure"] * features["MonthlyCharges"]
        labels = [0, 1, 0, 1, 0, 1, 0, 1]
        validation = features.iloc[:2].copy()
        validation["Contract"] = "Unseen"
        validation["MonthlyCharges"] = 10000

        for name, pipeline in baseline_models().items():
            with self.subTest(model=name):
                pipeline.fit(features, labels)
                preprocessing = pipeline.named_steps["preprocessing"]
                encoder = preprocessing.named_transformers_["categorical"]
                before = pipeline.predict_proba(features)
                added_fields = features.assign(customerID="ignored", gender="ignored", SeniorCitizen=999)
                np.testing.assert_array_equal(before, pipeline.predict_proba(added_fields))
                self.assertEqual(pipeline.predict_proba(validation).shape, (2, 2))
                contract_index = CATEGORICAL_COLUMNS.index("Contract")
                self.assertEqual(list(encoder.categories_[contract_index]), ["Seen"])

                transformed = preprocessing.transform(features)
                if name in ["KNN", "Logistic regression"]:
                    scaler = preprocessing.named_transformers_["numeric"]
                    np.testing.assert_allclose(scaler.mean_, features[NUMERIC_COLUMNS].mean())
                    self.assertEqual(scaler.n_samples_seen_, len(features))
                    np.testing.assert_allclose(transformed[:, :3].mean(axis=0), 0, atol=1e-12)
                else:
                    np.testing.assert_array_equal(transformed[:, :3], features[NUMERIC_COLUMNS])


if __name__ == "__main__":
    unittest.main()
