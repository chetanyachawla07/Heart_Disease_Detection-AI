"""Training pipeline.

    python train.py            # full run (a few minutes on a laptop)
    python train.py --fast     # tiny grids, ~20 s, used by CI

What it does, and why it is more trustworthy than a single train/test split:

1. Loads + cleans the data (fixes the inverted label, see ``data.py``).
2. Compares candidate models with *nested* repeated cross-validation: the inner
   loop tunes hyper-parameters, the outer loop measures performance on rows the
   tuning never saw. Reported numbers are mean ± std over outer folds.
3. Picks a model with the one-standard-error rule (simplest model that is
   statistically tied with the best).
4. Chooses an operating threshold from out-of-fold predictions so that recall
   (catching real disease) is at least ``TARGET_RECALL``.
5. Refits the chosen model on all rows and saves ONE artifact that contains the
   whole pipeline, threshold and metadata.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold

from . import __version__, reporting
from .data import DEFAULT_DATA_PATH, REPO_ROOT, file_sha256, load_dataset
from .modeling import baseline, candidates, select_model
from .schema import FEATURE_NAMES, FEATURES, NUMERIC_COLS

TARGET_RECALL = 0.90
ARTIFACT_FORMAT = 1
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "heart_model.joblib"
DEFAULT_OUT_DIR = REPO_ROOT / "outputs"


def _log(msg: str = "") -> None:
    print(msg, flush=True)


def _fold_metrics(y_true, proba) -> dict[str, float]:
    pred = (proba >= 0.5).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "avg_precision": average_precision_score(y_true, proba),
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "kappa": cohen_kappa_score(y_true, pred),
        "mcc": matthews_corrcoef(y_true, pred),
        "brier": brier_score_loss(y_true, proba),
    }


def choose_threshold(y, proba, target_recall: float = TARGET_RECALL) -> float:
    """Highest threshold that still reaches ``target_recall`` (max specificity)."""
    fpr, tpr, thr = roc_curve(y, proba)
    ok = np.where(tpr >= target_recall)[0]
    i = ok[np.argmin(fpr[ok])]
    t = float(thr[i])
    return float(np.clip(t, 0.01, 0.99))  # roc_curve can return inf / >1


def nested_cv(X, y, seed: int, fast: bool, log=_log):
    n_splits, n_repeats = (3, 1) if fast else (5, 2)
    inner_splits = 3 if fast else 5
    outer = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    folds = list(outer.split(X, y))
    models = {"Majority baseline": (baseline(), None), **candidates(seed, fast)}

    rows: dict[str, list[dict]] = {m: [] for m in models}
    oof = {m: np.zeros((n_repeats, len(y))) for m in models}
    fold_models: dict[str, list] = {m: [] for m in models}

    for name, (pipe, grid) in models.items():
        t0 = time.time()
        for k, (tr, te) in enumerate(folds):
            Xtr, ytr, Xte, yte = X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te]
            if grid is None:
                est = clone(pipe).fit(Xtr, ytr)
            else:
                inner = StratifiedKFold(inner_splits, shuffle=True, random_state=seed)
                est = GridSearchCV(pipe, grid, cv=inner, scoring="roc_auc", n_jobs=-1)
                est.fit(Xtr, ytr)
                est = est.best_estimator_
            proba = est.predict_proba(Xte)[:, 1]
            oof[name][k // n_splits, te] = proba
            rows[name].append(_fold_metrics(yte, proba))
            fold_models[name].append((est, te))
        log(f"  {name:22s} done in {time.time() - t0:5.1f}s")

    per_fold = {m: pd.DataFrame(r) for m, r in rows.items()}
    metric_names = list(next(iter(per_fold.values())).columns)
    summary = pd.DataFrame({
        **{f"{c}_mean": {m: df[c].mean() for m, df in per_fold.items()} for c in metric_names},
        **{f"{c}_std": {m: df[c].std(ddof=1) for m, df in per_fold.items()} for c in metric_names},
    })
    oof_mean = {m: a.mean(axis=0) for m, a in oof.items()}
    return summary, oof_mean, fold_models, len(folds)


def run_training(
    data_path=DEFAULT_DATA_PATH,
    out_dir=DEFAULT_OUT_DIR,
    model_path=DEFAULT_MODEL_PATH,
    seed: int = 42,
    fast: bool = False,
    log=_log,
) -> dict:
    out_dir, model_path = Path(out_dir), Path(model_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    log("STEP 1/6  Loading and cleaning data")
    X, y, report = load_dataset(data_path)
    df = X.assign(disease=y)
    log(f"  {report.rows_clean} patients ({report.positives} with disease); "
        f"dropped {report.duplicates_dropped} duplicate, nulled invalid codes "
        f"{report.invalid_codes_set_missing}")
    plots = reporting.plot_eda(df, out_dir)

    log("\nSTEP 2/6  Nested cross-validation" + (" (fast mode)" if fast else ""))
    summary, oof, fold_models, n_folds = nested_cv(X, y, seed, fast, log)
    summary = summary.sort_values("roc_auc_mean", ascending=False)
    shown = summary[["roc_auc_mean", "roc_auc_std", "recall_mean", "precision_mean",
                     "accuracy_mean", "kappa_mean", "mcc_mean", "brier_mean"]].round(3)
    log("\n" + shown.to_string())

    log("\nSTEP 3/6  Selecting a model")
    chosen, reason = select_model(summary, n_folds)
    log(f"  -> {chosen}. {reason}")

    log("\nSTEP 4/6  Choosing the decision threshold")
    p = oof[chosen]
    threshold = choose_threshold(y, p)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    at_thr = {
        "threshold": threshold, "recall": tp / (tp + fn), "specificity": tn / (tn + fp),
        "precision": tp / (tp + fp), "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }
    log(f"  threshold {threshold:.3f}: recall {at_thr['recall']:.2f}, "
        f"specificity {at_thr['specificity']:.2f}, precision {at_thr['precision']:.2f}")

    log("\nSTEP 5/6  Final fit on all patients + permutation importance")
    pipe, grid = candidates(seed, fast)[chosen]
    inner = StratifiedKFold(3 if fast else 5, shuffle=True, random_state=seed)
    search = GridSearchCV(pipe, grid, cv=inner, scoring="roc_auc", n_jobs=-1).fit(X, y)
    final = search.best_estimator_
    log(f"  best params: {search.best_params_}")

    imps = []
    for est, te in fold_models[chosen]:
        r = permutation_importance(est, X.iloc[te], y.iloc[te], scoring="roc_auc",
                                   n_repeats=3 if fast else 5, random_state=seed)
        imps.append(r.importances_mean)
    imps = np.array(imps)
    importance = pd.DataFrame({"mean": imps.mean(0), "std": imps.std(0)}, index=FEATURE_NAMES)

    plots += [
        reporting.plot_model_comparison(summary, out_dir),
        reporting.plot_roc(y, oof, chosen, out_dir),
        reporting.plot_confusion(y, p, threshold, chosen, out_dir),
        reporting.plot_calibration(y, p, chosen, out_dir),
        reporting.plot_threshold_tradeoff(y, p, threshold, out_dir),
        reporting.plot_permutation_importance(importance, out_dir),
    ]

    log("\nSTEP 6/6  Saving")
    reference = {
        f.name: float(X[f.name].median()) if f.name in NUMERIC_COLS or f.kind == "number"
        else float(X[f.name].mode().iloc[0])
        for f in FEATURES
    }
    training_ranges = {f.name: [float(X[f.name].min()), float(X[f.name].max())]
                       for f in FEATURES if f.kind == "number"}
    chosen_row = summary.loc[chosen]
    cv_metrics = {k.removesuffix("_mean"): float(v) for k, v in chosen_row.items()
                  if k.endswith("_mean")}
    cv_metrics.update({k: float(v) for k, v in chosen_row.items() if k.endswith("_std")})

    artifact = {
        "format_version": ARTIFACT_FORMAT,
        "pipeline": final,
        "model_name": chosen,
        "best_params": {k: (v if isinstance(v, (int, float, str, type(None))) else str(v))
                        for k, v in search.best_params_.items()},
        "threshold": threshold,
        "operating_point": {k: float(v) for k, v in at_thr.items()},
        "feature_order": FEATURE_NAMES,
        "reference": reference,
        "training_ranges": training_ranges,
        "cv_metrics": cv_metrics,
        "n_patients": int(len(y)),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "project_version": __version__,
    }
    joblib.dump(artifact, model_path, compress=3)

    summary.round(4).rename_axis("model").to_csv(out_dir / "model_comparison.csv")
    metrics = {
        "project_version": __version__,
        "seed": seed,
        "fast_mode": fast,
        "data_sha256": file_sha256(data_path),
        "cleaning": report.to_dict(),
        "cv": {"outer_folds": n_folds, "metric_note": "mean/std over outer folds, "
               "hyper-parameters tuned in an inner loop; threshold-based metrics at 0.5"},
        "models": json.loads(summary.round(4).to_json(orient="index")),
        "chosen_model": chosen,
        "selection_reason": reason,
        "best_params": artifact["best_params"],
        "operating_point_oof": {k: round(v, 4) if isinstance(v, float) else v
                                for k, v in at_thr.items()},
        "operating_point_note": "Threshold was chosen on the same out-of-fold predictions "
                                "it is reported on, so recall here is slightly optimistic.",
        "permutation_importance": json.loads(importance.round(4).to_json(orient="index")),
        "trained_at": artifact["trained_at"],
        "sklearn_version": sklearn.__version__,
        "plots": plots,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    log(f"  model   -> {model_path}")
    log(f"  reports -> {out_dir}/ (metrics.json, model_comparison.csv, {len(plots)} plots)")
    log("\nDone.")
    return metrics


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train the heart disease model.")
    ap.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fast", action="store_true", help="tiny grids, for smoke tests / CI")
    a = ap.parse_args(argv)
    run_training(a.data, a.out_dir, a.model_path, a.seed, a.fast)
    return 0


if __name__ == "__main__":
    sys.exit(main())
