"""
TEMPLATE model script. Copy this file into your own folder, rename it,
then change MODEL_NAME, the model and the grid. Run from the repo root:
    python modules/shared/template_train_model.py
"""
import sys, pathlib  # the 2-line fix: put the repo root on the import path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from sklearn.linear_model import LogisticRegression  # <- change
from sklearn.model_selection import GridSearchCV

from modules.shared.utils import (RANDOM_STATE, CV, SCORING, build_pipeline,
                                  get_data, save_results)

MODEL_NAME = "template_demo"  # <- change, e.g. "random_forest" (becomes results/<name>.json)

X_train, X_test, y_train, y_test = get_data()

# 1. Baseline: default hyperparameters, same pipeline (scaler -> SMOTE -> model)
baseline_pipe = build_pipeline(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))

# 2. Grid: every key needs the "model__" prefix because the model is a pipeline step
param_grid = {
    "model__C": [0.1, 1.0],
}

# 3. Tune on CV only (the test set is NOT touched here)
search = GridSearchCV(build_pipeline(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
                      param_grid, cv=CV, scoring=SCORING, refit="f1", n_jobs=-1)
search.fit(X_train, y_train)

# 4. Baseline vs tuned, CV + final test metrics -> results/<MODEL_NAME>.json
save_results(MODEL_NAME, baseline_pipe, search, X_train, y_train, X_test, y_test)
