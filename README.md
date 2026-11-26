# CM2604 Machine Learning Coursework

Binary classification of telecommunications customer churn using KNN, logistic regression and random forest.

## Project structure

| Folder | Contents |
|---|---|
| src | Dataset preparation, exploratory analysis and model evaluation scripts |
| tests | Checks for metric calculations and preprocessing |
| results | Exploratory figure and aggregate baseline results |
| data | Local copy of the source dataset, excluded from Git |

## Setup

Use Python 3.13. From the project directory in PowerShell, create a virtual environment and install the dependencies.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Download version 1 of BlastChar's Telco Customer Churn dataset and extract `WA_Fn-UseC_-Telco-Customer-Churn.csv` into `data/`. The CSV has 7,043 rows and 21 columns. Its SHA-256 checksum is `88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a`. [Source dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Run the analysis

```powershell
.\.venv\Scripts\python.exe src/eda.py
.\.venv\Scripts\python.exe -W error -m unittest discover -s tests -v
.\.venv\Scripts\python.exe src/baseline.py
.\.venv\Scripts\python.exe src/tune.py
.\.venv\Scripts\python.exe src/evaluate.py
.\.venv\Scripts\python.exe src/audit.py
```

The exploratory analysis writes `results/training-eda.png`. The baseline comparison writes `results/baseline.json`, including settings, fold scores and software versions. Both scripts resolve paths from their source location.

The tuning script searches 39 parameter combinations on the development folds and writes `results/tuning.json`. Run it before evaluation. The evaluation script fits the fixed baseline and selected settings on the development sample and writes `results/test.json` and `results/test-evaluation.png`. Customer-level test predictions are saved locally in `data/test_predictions.npz` for verification and the subgroup audit.

## Experimental method

The pipeline maps churn to 1 and retention to 0. It reserves a stratified 20% test sample with seed 42. Baseline estimates use five shuffled stratified folds on the remaining 80% and the same folds for every classifier. Scaling and one-hot encoding are learned inside each fitting fold. Numeric inputs are standardised for KNN and logistic regression.

Customer ID, gender and senior-citizen status are excluded from fitting. The demographic fields are retained locally for the subgroup audit. Eleven blank total-charge values occur at zero tenure. Their cumulative charges are estimated as zero under a fixed preparation rule.

Average precision is the primary comparison metric. Precision, recall, F1 and accuracy use a churn probability threshold of 0.50. Hyperparameters are selected using the same development folds as the baseline experiment. The baseline appears first in each grid so it is retained on an exact tie. A random forest with 300 trees and minimum leaf size 15 was nominated using development average precision before test evaluation. Logistic regression retained C = 1. KNN selected 31 uniformly weighted neighbours and Euclidean distance.

## Held-out results

The test sample contains 1,409 customers, including 374 churners. These results compare the six fixed settings.

| Model | Setting | Average precision | ROC AUC | Precision | Recall | F1 | Accuracy |
|---|---|---:|---:|---:|---:|---:|---:|
| KNN | Baseline | 0.506 | 0.778 | 0.542 | 0.500 | 0.520 | 0.755 |
| KNN | Tuned | 0.606 | 0.829 | 0.593 | 0.564 | 0.578 | 0.781 |
| Logistic regression | Baseline | 0.632 | 0.841 | 0.650 | 0.551 | 0.596 | 0.802 |
| Logistic regression | Tuned | 0.632 | 0.841 | 0.650 | 0.551 | 0.596 | 0.802 |
| Random forest | Baseline | 0.609 | 0.818 | 0.614 | 0.511 | 0.558 | 0.785 |
| Random forest | Tuned | 0.652 | 0.843 | 0.679 | 0.497 | 0.574 | 0.804 |

The nominated forest achieved the highest test average precision. Logistic regression achieved higher F1 and recall at the fixed threshold, which shows the trade-off between ranking quality and a particular contact rule. Its baseline and tuned results are identical because the search retained its baseline setting. The development AP difference between the forest and logistic regression was only 0.004, so this experiment supports a cautious comparison.

![Test ROC curves, precision-recall curves and nominated-model confusion matrix](results/test-evaluation.png)

## Ethics and subgroup errors

The development process excludes customer identifiers from fitting, keeps source data and customer-level predictions outside Git, and publishes aggregate results. Gender and senior-citizen status are retained for auditing and excluded from predictors. Other attributes may act as proxies, so this exclusion alone cannot demonstrate fairness. Reproducible experiments, an independent test sample and reporting of missed churners support honest performance claims. The BCS Code of Conduct connects this work to privacy, non-discrimination and professional integrity (BCS, n.d.).

Run `src/audit.py` after evaluation to produce `results/subgroups.json`. It uses the saved test predictions for the nominated random forest at the original 0.50 threshold. Recall divides correctly identified churners by actual churners. The false-positive rate divides incorrectly flagged customers by actual customers who stayed.

| Group | Customers | Recall, identified / churned | False-positive rate, flagged / stayed |
|---|---:|---:|---:|
| Female | 687 | 49.2% (95/193) | 7.5% (37/494) |
| Male | 722 | 50.3% (91/181) | 9.4% (51/541) |
| SeniorCitizen = 0 | 1,187 | 48.6% (134/276) | 7.1% (65/911) |
| SeniorCitizen = 1 | 222 | 53.1% (52/98) | 18.5% (23/124) |

The senior group's false-positive rate is 11.4 percentage points higher. The estimate is based on 124 customers who stayed, so sampling variation and group composition need investigation. These observations do not establish the cause of the difference or certify fairness. The audit covers the recorded binary demographic categories separately. A wider population and combinations of characteristics require further evidence.

For a real retention service, missed churners could lose access to helpful offers, while false positives could receive unwanted contact. Human review, understandable explanations and a way to correct data or decline contact should accompany any pilot. Use the ICO's privacy principles as a design reference, and determine the applicable jurisdiction, lawful basis, source permissions and customer rights before processing real customer data (ICO, n.d.).

## Proposed post-deployment checks

The following is a proposed operational plan. It adapts the NIST framework's governance, measurement and risk-management functions to this scenario (NIST, n.d.).

| Check | Proposed owner and cadence | Response |
|---|---|---|
| Input validity, missing values, unseen categories and changing customer mix | Data owner, weekly | Investigate pipeline failures or material shifts and pause affected predictions when inputs are unreliable. |
| AP, precision, recall, F1 and confusion counts, overall and by group | Model owner, monthly once churn outcomes are available | Compare with validated pilot ranges, review recurring group gaps and escalate deteriorating performance. Set alert limits during the pilot. |
| Complaints, unwanted contact and unequal access to offers | Customer-service lead, ongoing | Review individual cases, correct errors and adjust the outreach process. |
| Access permissions, retention and model changes | Privacy lead and model owner, periodically and before each release | Restrict access, apply the agreed retention policy, validate replacements on fresh data and retain an approved rollback option. |

## Limitations and future improvements

- The single historical dataset and random split provide a retrospective benchmark. Validate feature availability before prediction and use a later, representative customer cohort before operational use.
- The fixed 0.50 threshold misses 188 of 374 churners for the nominated model. Assess contact costs, capacity and customer impact in a fresh validation study before choosing an operational threshold.
- The 11 zero-tenure charge replacements are assumptions. Verify billing semantics with the data owner and test sensitivity to alternative treatments on development data.
- The small development AP difference between forest and logistic regression and the partial subgroup audit limit a strong model preference. Use additional customer samples to assess performance stability and error differences.
- Churn prediction alone does not show that contacting a customer prevents cancellation. A controlled retention pilot should measure whether the intervention helps customers and improves retention.

## Method references

BCS (n.d.) *BCS Code of Conduct*. Available at: https://www.bcs.org/membership-and-registrations/become-a-member/bcs-code-of-conduct/ (Accessed: 25 November 2026).

Information Commissioner's Office (ICO) (n.d.) *A guide to the data protection principles*. Available at: https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/data-protection-principles/a-guide-to-the-data-protection-principles/ (Accessed: 25 November 2026).

National Institute of Standards and Technology (NIST) (n.d.) *NIST AI RMF Playbook*. Available at: https://airc.nist.gov/airmf-resources/playbook/ (Accessed: 25 November 2026).

scikit-learn developers (n.d.) *GridSearchCV*. Available at: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html (Accessed: 23 November 2026).
