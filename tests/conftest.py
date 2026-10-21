import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heart_disease.predictor import Predictor  # noqa: E402
from heart_disease.train import run_training  # noqa: E402

# Two patients no clinician would confuse. Used as direction-of-risk regression tests.
HIGH_RISK = dict(age=65, sex=1, cp=0, trestbps=160, chol=280, fbs=1, restecg=1, thalach=105,
                 exang=1, oldpeak=3.5, slope=1, ca=3, thal=3)
LOW_RISK = dict(age=35, sex=0, cp=1, trestbps=115, chol=180, fbs=0, restecg=0, thalach=185,
                exang=0, oldpeak=0.0, slope=2, ca=0, thal=2)


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    """Train once in --fast mode into a temp dir (~10 s); shared by all tests."""
    d = tmp_path_factory.mktemp("train")
    model = d / "model.joblib"
    metrics = run_training(out_dir=d / "out", model_path=model, fast=True, log=lambda *_: None)
    return {"model_path": model, "out_dir": d / "out", "metrics": metrics}


@pytest.fixture(scope="session")
def predictor(trained):
    return Predictor.load(trained["model_path"])
