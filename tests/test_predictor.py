import contextlib
import io
import json
import sys
from pathlib import Path

import pytest

from heart_disease.predictor import DEFAULT_MODEL_PATH, Predictor, ValidationError
from heart_disease.schema import EXAMPLE_PATIENT

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conftest import HIGH_RISK, LOW_RISK  # noqa: E402


def test_risk_direction_is_correct(predictor):
    """The original app scored an obviously sick patient at 9.5% and a healthy one at 99.5%."""
    high, low = predictor.predict(HIGH_RISK), predictor.predict(LOW_RISK)
    assert high.probability > 0.8 and high.flagged
    assert low.probability < 0.2 and not low.flagged
    assert high.label == "Higher risk indicated"


def test_more_ischaemia_signs_means_higher_risk(predictor):
    base = predictor.predict(LOW_RISK).probability
    worse = predictor.predict({**LOW_RISK, "exang": 1, "oldpeak": 3.0, "ca": 2}).probability
    assert worse > base


def test_explanation_structure(predictor):
    result = predictor.predict(HIGH_RISK)
    assert 1 <= len(result.factors) <= 5
    effects = [abs(f.log_odds) for f in result.factors]
    assert effects == sorted(effects, reverse=True)
    assert any(f.to_dict()["direction"] == "raises" for f in result.factors)
    assert predictor.predict(HIGH_RISK, explain=False).factors == []


def test_out_of_training_range_warns(predictor):
    result = predictor.predict({**LOW_RISK, "age": 95})
    assert any("Age" in w for w in result.warnings)
    assert predictor.predict(LOW_RISK).warnings == []


def test_invalid_input_raises(predictor):
    with pytest.raises(ValidationError):
        predictor.predict({**LOW_RISK, "cp": 7})


def test_missing_model_file_has_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="python train.py"):
        Predictor.load(tmp_path / "nope.joblib")


def test_cli_json_output(trained):
    import predict

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = predict.main(["--model", str(trained["model_path"]), "--json", "--age", "40"])
    out = json.loads(buf.getvalue())
    assert code == 0 and out["patient"]["age"] == 40 and "probability" in out


def test_cli_rejects_bad_input(trained):
    import predict

    assert predict.main(["--model", str(trained["model_path"]), "--age", "999"]) == 2


def test_committed_model_matches_expectations():
    if not DEFAULT_MODEL_PATH.exists():
        pytest.skip("no committed model; run `python train.py`")
    p = Predictor.load()
    assert p.predict(HIGH_RISK).probability > p.predict(LOW_RISK).probability
    assert p.predict(EXAMPLE_PATIENT).probability is not None
