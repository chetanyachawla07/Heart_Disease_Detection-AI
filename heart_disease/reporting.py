"""Plots written to outputs/. Matplotlib runs headless (Agg)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve  # noqa: E402

from .data import LABEL  # noqa: E402
from .schema import FEATURES_BY_NAME  # noqa: E402

DPI = 130


def _save(fig, path: Path) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path.name


def plot_eda(df: pd.DataFrame, out: Path) -> list[str]:
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title(f"Feature correlation heatmap (label = '{LABEL}', 1 = disease)")
    a = _save(fig, out / "01_correlation_heatmap.png")

    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df[LABEL].value_counts().sort_index()
    ax.bar(["No disease", "Disease"], counts.values, color=["#3A7D44", "#D1495B"])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 2, str(v), ha="center")
    ax.set_title("Class balance (after cleaning)")
    b = _save(fig, out / "02_class_balance.png")
    return [a, b]


def plot_model_comparison(summary: pd.DataFrame, out: Path) -> str:
    s = summary.sort_values("roc_auc_mean")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(s.index, s["roc_auc_mean"], xerr=s["roc_auc_std"], color="#2E8B8B", capsize=4)
    ax.axvline(0.5, color="grey", ls="--", lw=1)
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel("ROC-AUC (nested CV, mean ± std over outer folds)")
    ax.set_title("Model comparison")
    return _save(fig, out / "03_model_comparison.png")


def plot_roc(y, oof: dict[str, np.ndarray], chosen: str, out: Path) -> str:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for name, p in oof.items():
        if name == "Majority baseline":
            continue
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, lw=2.5 if name == chosen else 1, label=name,
                alpha=1 if name == chosen else 0.6)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate",
           title="ROC curves (out-of-fold predictions)")
    ax.legend(loc="lower right")
    return _save(fig, out / "04_roc_curve.png")


def plot_confusion(y, p, threshold: float, name: str, out: Path) -> str:
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ConfusionMatrixDisplay.from_predictions(
        y, (p >= threshold).astype(int), display_labels=["No disease", "Disease"],
        cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(f"{name}: out-of-fold, threshold {threshold:.2f}")
    return _save(fig, out / "05_confusion_matrix.png")


def plot_calibration(y, p, name: str, out: Path) -> str:
    frac, mean_pred = calibration_curve(y, p, n_bins=8, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfectly calibrated")
    ax.plot(mean_pred, frac, "o-", color="#D1495B", label=name)
    ax.set(xlabel="Predicted probability", ylabel="Observed fraction with disease",
           title="Calibration (out-of-fold)")
    ax.legend()
    return _save(fig, out / "06_calibration.png")


def plot_threshold_tradeoff(y, p, threshold: float, out: Path) -> str:
    ts = np.linspace(0.05, 0.95, 91)
    y = np.asarray(y)
    rec = [((p >= t) & (y == 1)).sum() / (y == 1).sum() for t in ts]
    spec = [((p < t) & (y == 0)).sum() / (y == 0).sum() for t in ts]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ts, rec, label="Recall (sensitivity)", color="#D1495B")
    ax.plot(ts, spec, label="Specificity", color="#3A7D44")
    ax.axvline(threshold, color="k", ls="--", lw=1, label=f"Chosen threshold {threshold:.2f}")
    ax.set(xlabel="Decision threshold", ylabel="Rate", title="Threshold trade-off")
    ax.legend()
    return _save(fig, out / "07_threshold_tradeoff.png")


def plot_permutation_importance(imp: pd.DataFrame, out: Path) -> str:
    imp = imp.sort_values("mean")
    labels = [FEATURES_BY_NAME[f].label for f in imp.index]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.barh(labels, imp["mean"], xerr=imp["std"], color="#2E8B8B", capsize=3)
    ax.set_xlabel("Drop in ROC-AUC when the feature is shuffled (held-out folds)")
    ax.set_title("Permutation importance")
    return _save(fig, out / "08_permutation_importance.png")
