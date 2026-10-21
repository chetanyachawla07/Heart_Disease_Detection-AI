import pytest

pytest.importorskip("flask")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conftest import HIGH_RISK, LOW_RISK  # noqa: E402

from webapp.app import create_app  # noqa: E402


@pytest.fixture(scope="module")
def client(predictor):
    app = create_app(predictor=predictor)
    app.testing = True
    return app.test_client()


def test_home_page_renders_form_from_schema(client):
    html = client.get("/").get_data(as_text=True)
    assert 'name="age"' in html and 'name="thal"' in html
    assert "Fixed defect" in html and "Reversible defect" in html
    assert "Protype" not in html


def test_security_headers(client):
    r = client.get("/")
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert "default-src 'self'" in r.headers["Content-Security-Policy"]


def test_health_and_schema(client):
    assert client.get("/health").get_json()["status"] == "ok"
    feats = client.get("/api/schema").get_json()["features"]
    assert len(feats) == 13 and feats[0]["name"] == "age"


def test_predict_high_and_low_risk(client):
    hi = client.post("/predict", json=HIGH_RISK).get_json()
    lo = client.post("/predict", json=LOW_RISK).get_json()
    assert hi["flagged"] and hi["probability"] > 80 and hi["prediction"] == 1
    assert not lo["flagged"] and lo["probability"] < 20 and lo["prediction"] == 0
    assert {"label", "probability", "threshold", "factors", "warnings"} <= set(hi)


def test_predict_reports_field_errors(client):
    r = client.post("/predict", json={**LOW_RISK, "age": 999, "cp": 9})
    body = r.get_json()
    assert r.status_code == 400 and set(body["fields"]) == {"age", "cp"}


def test_predict_rejects_non_json_and_wrong_shapes(client):
    assert client.post("/predict", data="age=1").status_code == 400
    assert client.post("/predict", json=[1, 2, 3]).status_code == 400
    assert client.post("/predict", json={}).status_code == 400
    bad = client.post("/predict", data="{not json", content_type="application/json")
    assert bad.status_code == 400


def test_predict_rejects_oversized_body(client):
    assert client.post("/predict", json={"x": "a" * 40000}).status_code == 413


def test_predict_only_accepts_post(client):
    assert client.get("/predict").status_code == 405
