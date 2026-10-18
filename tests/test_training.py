import json

import numpy as np
import pandas as pd

from heart_disease.modeling import select_model
from heart_disease.train import TARGET_RECALL, choose_threshold


def test_training_outputs_exist(trained):
    out = trained["out_dir"]
    assert trained["model_path"].exists()
    assert (out / "metrics.json").exists()
    assert (out / "model_comparison.csv").exists()
    pngs = sorted(p.name for p in out.glob("*.png"))
    assert len(pngs) == 8, pngs


def test_model_beats_baseline_by_a_wide_margin(trained):
    models = trained["metrics"]["models"]
    assert models["Majority baseline"]["roc_auc_mean"] == 0.5
    chosen = trained["metrics"]["chosen_model"]
    assert models[chosen]["roc_auc_mean"] > 0.8


def test_naive_bayes_kappa_and_mcc_are_reported(trained):
    models = trained["metrics"]["models"]
    assert "Gaussian Naive Bayes" in models
    for m in models.values():
        assert "kappa_mean" in m and "mcc_mean" in m
    assert models["Majority baseline"]["mcc_mean"] == 0.0


def test_metrics_json_is_valid_and_complete(trained):
    data = json.loads((trained["out_dir"] / "metrics.json").read_text())
    for key in ("chosen_model", "selection_reason", "operating_point_oof",
                "permutation_importance", "cleaning", "data_sha256"):
        assert key in data
    assert data["operating_point_oof"]["recall"] >= TARGET_RECALL - 0.02


def test_choose_threshold_meets_target_recall():
    rng = np.random.default_rng(0)
    y = pd.Series(rng.integers(0, 2, 400))
    p = np.clip(0.5 * y + rng.normal(0.25, 0.2, 400), 0, 1)
    t = choose_threshold(y, p)
    recall = ((p >= t) & (y == 1)).sum() / (y == 1).sum()
    assert 0 < t < 1 and recall >= TARGET_RECALL


def test_one_standard_error_rule_prefers_simpler_model():
    summary = pd.DataFrame(
        {"roc_auc_mean": [0.90, 0.905, 0.80, 0.5], "roc_auc_std": [0.05, 0.05, 0.05, 0.0]},
        index=["Logistic Regression", "Random Forest", "Gradient Boosting", "Majority baseline"],
    )
    name, reason = select_model(summary, n_folds=10)
    assert name == "Logistic Regression" and "simpler" in reason

    summary.loc["Random Forest", "roc_auc_mean"] = 0.95   # clearly better now
    assert select_model(summary, n_folds=10)[0] == "Random Forest"
