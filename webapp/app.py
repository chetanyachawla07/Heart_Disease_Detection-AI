"""Flask web app for the Heart Disease Risk Predictor.

    python webapp/app.py                  # development server on http://localhost:5000
    gunicorn "webapp.app:create_app()"    # production-style (see Dockerfile)

Environment variables: HEART_MODEL_PATH, HOST (default 127.0.0.1), PORT (default 5000).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow ``python webapp/app.py`` from any directory (the package lives one level up).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flask import Flask, jsonify, render_template, request  # noqa: E402

from heart_disease.predictor import Predictor, ValidationError  # noqa: E402
from heart_disease.schema import FEATURES  # noqa: E402

CSP = (
    "default-src 'self'; style-src 'self' https://fonts.googleapis.com; "
    "font-src https://fonts.gstatic.com; img-src 'self' data:; script-src 'self'; "
    "base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
)


def create_app(predictor: Predictor | None = None, model_path: str | None = None) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024  # a patient record is ~300 bytes
    predictor = predictor or Predictor.load(model_path or os.environ.get("HEART_MODEL_PATH"))
    summary = predictor.summary()

    @app.after_request
    def security_headers(resp):
        resp.headers.setdefault("Content-Security-Policy", CSP)
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("Referrer-Policy", "no-referrer")
        return resp

    @app.get("/")
    def home():
        return render_template("index.html", features=FEATURES, info=summary)

    @app.get("/health")
    def health():
        return jsonify(status="ok", model=summary["model_name"])

    @app.get("/api/schema")
    def schema():
        return jsonify(features=[f.to_dict() for f in FEATURES])

    @app.post("/predict")
    def predict():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify(error="Send a JSON object with the patient's measurements."), 400
        try:
            result = predictor.predict(data)
        except ValidationError as exc:
            return jsonify(error="Some fields are invalid.", fields=exc.errors), 400
        return jsonify(result.to_dict())

    @app.errorhandler(413)
    def too_large(_):
        return jsonify(error="Request too large."), 413

    return app


if __name__ == "__main__":
    create_app().run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "5000")),
        debug=False,
    )
