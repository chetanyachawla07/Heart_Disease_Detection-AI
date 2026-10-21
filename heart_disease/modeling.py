"""Model pipelines, candidate models and the model-selection rule."""

from __future__ import annotations

import math

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from .schema import BINARY_COLS, NOMINAL_COLS, NUMERIC_COLS


def build_pipeline(estimator) -> Pipeline:
    """Preprocessing + estimator in ONE object.

    Imputation, scaling and encoding are fitted inside each cross-validation
    fold, so nothing leaks from validation rows into training. The same object
    is saved and used at prediction time, so train/serve preprocessing cannot
    diverge.
    """
    pre = ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), NUMERIC_COLS),
            ("bin", SimpleImputer(strategy="most_frequent"), BINARY_COLS),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                              ("onehot", OneHotEncoder(handle_unknown="ignore"))]),
             NOMINAL_COLS),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline([("prep", pre), ("clf", estimator)])


# Lower rank = simpler / more interpretable. Used by the selection rule below.
COMPLEXITY = {
    "Majority baseline": 0,
    "Logistic Regression": 1,
    "Gaussian Naive Bayes": 1.5,
    "SVM (RBF)": 2,
    "Random Forest": 3,
    "Gradient Boosting": 4,
}


def candidates(seed: int, fast: bool = False) -> dict[str, tuple[Pipeline, dict]]:
    """name -> (pipeline, hyper-parameter grid). ``fast`` shrinks everything for CI."""
    n_trees = 100 if fast else 200
    grids = {
        "Logistic Regression": {"clf__C": [0.1, 1] if fast else [0.03, 0.1, 0.3, 1, 3, 10]},
        "Gaussian Naive Bayes": {
            "clf__var_smoothing": [1e-9, 1e-6] if fast else [1e-9, 1e-7, 1e-5, 1e-3, 1e-1]
        },
        "SVM (RBF)": {
            "clf__C": [1, 3] if fast else [0.3, 1, 3, 10],
            "clf__gamma": ["scale"] if fast else ["scale", 0.01, 0.03],
        },
        "Random Forest": {
            "clf__max_depth": [3, None] if fast else [3, 5, None],
            "clf__min_samples_leaf": [3] if fast else [1, 3, 5],
        },
        "Gradient Boosting": {
            "clf__n_estimators": [50] if fast else [50, 100, 200],
            "clf__learning_rate": [0.1] if fast else [0.03, 0.1],
            "clf__max_depth": [2] if fast else [1, 2, 3],
        },
    }
    estimators = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Gaussian Naive Bayes": GaussianNB(),
        "SVM (RBF)": SVC(probability=True, random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=n_trees, random_state=seed),
        "Gradient Boosting": GradientBoostingClassifier(random_state=seed),
    }
    return {name: (build_pipeline(est), grids[name]) for name, est in estimators.items()}


def baseline() -> Pipeline:
    return build_pipeline(DummyClassifier(strategy="prior"))


def select_model(summary: pd.DataFrame, n_folds: int) -> tuple[str, str]:
    """One-standard-error rule.

    With ~300 patients the AUC differences between good models are well inside
    the noise. So: find the best mean ROC-AUC, then choose the SIMPLEST model
    whose mean AUC is within one standard error of it. Returns (name, reason).
    ``summary`` is indexed by model name with ``roc_auc_mean`` / ``roc_auc_std``.
    """
    pool = summary.drop(index="Majority baseline", errors="ignore")
    best = pool["roc_auc_mean"].idxmax()
    se = pool.loc[best, "roc_auc_std"] / math.sqrt(n_folds)
    cutoff = pool.loc[best, "roc_auc_mean"] - se
    eligible = pool[pool["roc_auc_mean"] >= cutoff]
    chosen = min(eligible.index, key=lambda n: COMPLEXITY.get(n, 99))
    if chosen == best:
        reason = f"{chosen} has the highest mean ROC-AUC."
    else:
        reason = (f"{best} scored highest, but {chosen} is within one standard error "
                  f"({se:.3f}) and simpler, so it was preferred.")
    return chosen, reason
