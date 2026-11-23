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
```

The exploratory analysis writes `results/training-eda.png`. The baseline comparison writes `results/baseline.json`, including settings, fold scores and software versions. Both scripts resolve paths from their source location.

## Experimental method

The pipeline maps churn to 1 and retention to 0. It reserves a stratified 20% test sample with seed 42. Baseline estimates use five shuffled stratified folds on the remaining 80% and the same folds for every classifier. Scaling and one-hot encoding are learned inside each fitting fold. Numeric inputs are standardised for KNN and logistic regression.

Customer ID, gender and senior-citizen status are excluded from fitting. The demographic fields are retained locally for the subgroup audit. Eleven blank total-charge values occur at zero tenure. Their cumulative charges are estimated as zero under a fixed preparation rule.

Average precision is the primary comparison metric. Precision, recall, F1 and accuracy use a churn probability threshold of 0.50. Logistic regression has the highest baseline validation average precision. Hyperparameter tuning and held-out evaluation form the next experimental stage.
