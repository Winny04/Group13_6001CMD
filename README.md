# Telco Customer Churn Prediction – 6001CMD Group 13

Machine learning group assignment for 6001CMD (Coventry University / INTI International College Penang).

We predict whether a telecom customer will churn (supervised binary classification, Churn Yes = 1 / No = 0) using the [Kaggle Telco Customer Churn dataset](https://www.kaggle.com/blastchar/telco-customer-churn).

## Team

| Member | Responsibility |
|---|---|
| Boon | Problem definition, EDA, Logistic Regression |
| Chang | Data preparation pipeline, Random Forest, final notebook |
| Justin | Gradient Boosting, soft voting ensemble, evaluation |

## Results summary (tuned models)

| Model | CV F1 | Test F1 | Test Recall | Test ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.6333 | 0.6137 | 0.7581 | 0.8400 |
| Random Forest | 0.6371 | 0.6272 | 0.7124 | 0.8397 |
| Gradient Boosting | 0.6358 | 0.6290 | 0.7177 | 0.8400 |
| Soft Voting Ensemble (RF + GB) | 0.6345 | 0.6295 | 0.7124 | 0.8411 |

CV = 5-fold stratified cross-validation on the training set. The test set (1,405 rows) was used only for final reporting. Full discussion is in the report and in `notebooks/FINAL_SUBMISSION_NOTEBOOK.ipynb`.

## Repository structure

```
data/
  raw/                         original Kaggle CSV
  processed/                   cleaned and encoded data (not scaled, not resampled)
modules/
  shared/utils.py              shared split, pipeline (scaler -> SMOTE -> model), CV and save_results()
  01_eda_and_baseline/         EDA script and Logistic Regression
  02_preprocessing_pipeline/   data pipeline, pipeline tests, SMOTE impact experiment, correlation check
  03_tree_models_and_ensemble/ Random Forest, Gradient Boosting, ensemble and evaluation
notebooks/
  FINAL_SUBMISSION_NOTEBOOK.ipynb   full project, Parts A–G
  member1_boon.ipynb                EDA and Logistic Regression working notebook
results/                       saved metrics for every model (JSON)
main.py                        setup check
requirements.txt               pinned package versions
```

## How to run

Run every command from the repository root.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt

python main.py                                                   # setup check
python modules/02_preprocessing_pipeline/test_pipeline.py        # 7 pipeline checks
python modules/01_eda_and_baseline/run_eda.py                    # EDA figures
python modules/01_eda_and_baseline/train_logistic_regression.py
python modules/03_tree_models_and_ensemble/train_random_forest.py
python modules/03_tree_models_and_ensemble/train_gradient_boosting.py
python modules/03_tree_models_and_ensemble/run_ensemble.py
python modules/03_tree_models_and_ensemble/run_evaluation.py
```

To view everything in one place, open `notebooks/FINAL_SUBMISSION_NOTEBOOK.ipynb` with the venv kernel and choose Run All.

## Reproducibility notes

- `random_state = 42` is used for the split, SMOTE, cross-validation and all models.
- Scaling and SMOTE sit inside the model pipeline, so they are fitted on training folds only (no data leakage).
- Package versions are pinned in `requirements.txt`. Results can differ in the 3rd–4th decimal place between Python versions, so the report quotes values to 3–4 decimals.
