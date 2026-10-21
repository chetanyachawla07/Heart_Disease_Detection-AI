"""Loading the saved model, validating input, predicting and explaining.

Every front-end (CLI, Flask, Streamlit) goes through ``Predictor`` so they all
validate and behave identically.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import joblib
import pandas as pd
import sklearn

from .schema import FEATURE_NAMES, FEATURES, FEATURES_BY_NAME

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "heart_model.joblib"

MIN_FACTOR_LOG_ODDS = 0.1  # hide factors that change the odds by less than ~10%
MAX_FACTORS = 5


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


class ValidationError(ValueError):
    """Raised for bad patient input. ``errors`` maps field name -> message."""

    def __init__(self, errors: dict[str, str]):
        self.errors = errors
        super().__init__("; ".join(f"{k}: {v}" for k, v in errors.items()))


def validate_patient(raw: Mapping[str, Any]) -> dict[str, float]:
    """Check every field, returning clean numbers or raising ValidationError."""
    clean: dict[str, float] = {}
    errors: dict[str, str] = {}
    for f in FEATURES:
        if f.name not in raw or raw[f.name] in (None, ""):
            errors[f.name] = "This field is required."
            continue
        value = raw[f.name]
        if isinstance(value, bool):
            value = int(value)
        try:
            num = float(value)
        except (TypeError, ValueError):
            errors[f.name] = "Must be a number."
            continue
        if not math.isfinite(num):
            errors[f.name] = "Must be a finite number."
            continue
        if f.kind == "choice":
            if num != int(num) or int(num) not in f.choice_codes:
                errors[f.name] = f"Must be one of {list(f.choice_codes)}."
                continue
            clean[f.name] = int(num)
            continue
        if f.integer and num != int(num):
            errors[f.name] = "Must be a whole number."
            continue
        if num < f.min or num > f.max:
            errors[f.name] = f"Must be between {f.min:g} and {f.max:g} {f.unit}".strip() + "."
            continue
        clean[f.name] = int(num) if f.integer else num
    if errors:
        raise ValidationError(errors)
    return clean


@dataclass
class Factor:
    feature: str
    label: str
    value: str
    log_odds: float  # change in log-odds of disease vs. a typical patient

    @property
    def odds_ratio(self) -> float:
        return math.exp(self.log_odds)

    def to_dict(self) -> dict:
        return {
            "feature": self.feature, "label": self.label, "value": self.value,
            "odds_ratio": round(self.odds_ratio, 2),
            "direction": "raises" if self.log_odds > 0 else "lowers",
        }


@dataclass
class Prediction:
    probability: float  # 0-1
    flagged: bool
    threshold: float
    factors: list[Factor] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return "Higher risk indicated" if self.flagged else "Lower risk indicated"

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "flagged": self.flagged,
            "prediction": int(self.flagged),  # kept for backwards compatibility
            "probability": round(self.probability * 100, 1),  # percent
            "threshold": round(self.threshold * 100, 1),  # percent
            "factors": [f.to_dict() for f in self.factors],
            "warnings": self.warnings,
        }


class Predictor:
    def __init__(self, artifact: dict):
        self._a = artifact
        self.pipeline = artifact["pipeline"]
        self.threshold: float = artifact["threshold"]
        self.reference: dict[str, float] = artifact["reference"]
        self.training_ranges: dict[str, list[float]] = artifact["training_ranges"]

    @classmethod
    def load(cls, path: str | Path | None = None) -> Predictor:
        path = Path(path or DEFAULT_MODEL_PATH)
        if not path.exists():
            raise FileNotFoundError(
                f"No trained model at {path}. Create one with:  python train.py"
            )
        # joblib/pickle can execute code: only load model files you trained yourself.
        artifact = joblib.load(path)
        trained = artifact.get("sklearn_version")
        if trained and trained.split(".")[:2] != sklearn.__version__.split(".")[:2]:
            warnings.warn(
                f"Model was trained with scikit-learn {trained} but {sklearn.__version__} "
                "is installed; retrain with `python train.py` if results look wrong.",
                stacklevel=2,
            )
        return cls(artifact)

    # -- metadata shown in UIs ------------------------------------------------
    def summary(self) -> dict:
        m = self._a.get("cv_metrics", {})
        op = self._a.get("operating_point", {})
        return {
            "model_name": self._a["model_name"],
            "n_patients": self._a["n_patients"],
            "roc_auc": m.get("roc_auc"),
            "roc_auc_std": m.get("roc_auc_std"),
            "threshold": self.threshold,
            "recall": op.get("recall"),
            "specificity": op.get("specificity"),
            "trained_at": self._a["trained_at"],
        }

    # -- prediction -------------------------------------------------------------
    def _proba(self, rows: list[dict]) -> list[float]:
        df = pd.DataFrame(rows)[FEATURE_NAMES].astype(float)
        return [float(p) for p in self.pipeline.predict_proba(df)[:, 1]]

    def _range_warnings(self, patient: Mapping[str, float]) -> list[str]:
        out = []
        for name, (lo, hi) in self.training_ranges.items():
            v = patient[name]
            if v < lo or v > hi:
                f = FEATURES_BY_NAME[name]
                out.append(
                    f"{f.label} ({v:g}) is outside the range seen in the training data "
                    f"({lo:g}-{hi:g}); the estimate is less reliable."
                )
        return out

    def explain(self, patient: Mapping[str, float], base_prob: float) -> list[Factor]:
        """One-at-a-time what-if: swap each value for that of a typical patient
        (training median / most common code) and see how the odds of disease
        change. Odds, not percentage points, so it still works when the
        probability is close to 0% or 100%. Ignores interactions between
        features, so read it as a hint, not a proof."""
        variants = [{**patient, f.name: self.reference[f.name]} for f in FEATURES]
        probs = self._proba(variants)
        base = _logit(base_prob)
        factors = []
        for f, p_ref in zip(FEATURES, probs):
            delta = base - _logit(p_ref)
            if abs(delta) >= MIN_FACTOR_LOG_ODDS:
                factors.append(Factor(f.name, f.label, f.display_value(patient[f.name]), delta))
        factors.sort(key=lambda x: abs(x.log_odds), reverse=True)
        return factors[:MAX_FACTORS]

    def predict(self, raw: Mapping[str, Any], explain: bool = True) -> Prediction:
        patient = validate_patient(raw)
        prob = self._proba([patient])[0]
        return Prediction(
            probability=prob,
            flagged=prob >= self.threshold,
            threshold=self.threshold,
            factors=self.explain(patient, prob) if explain else [],
            warnings=self._range_warnings(patient),
        )
