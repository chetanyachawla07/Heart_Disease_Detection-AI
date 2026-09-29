"""
Flask web app for the Heart Disease Risk Predictor.
Run with:  python app.py
Then open: http://localhost:5000
"""
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

model = joblib.load(os.path.join(BASE_DIR, "models", "heart_disease_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))
feature_names = joblib.load(os.path.join(BASE_DIR, "models", "feature_names.pkl"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        patient = {
            "age": float(data["age"]),
            "sex": int(data["sex"]),
            "cp": int(data["cp"]),
            "trestbps": float(data["trestbps"]),
            "chol": float(data["chol"]),
            "fbs": int(data["fbs"]),
            "restecg": int(data["restecg"]),
            "thalach": float(data["thalach"]),
            "exang": int(data["exang"]),
            "oldpeak": float(data["oldpeak"]),
            "slope": int(data["slope"]),
            "ca": int(data["ca"]),
            "thal": int(data["thal"]),
        }
    except (KeyError, ValueError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    X_new = pd.DataFrame([patient])[feature_names]
    X_scaled = scaler.transform(X_new)

    prediction = int(model.predict(X_scaled)[0])
    probability = float(model.predict_proba(X_scaled)[0][1])

    return jsonify({
        "prediction": prediction,
        "probability": round(probability * 100, 1),
        "label": "Higher risk indicated" if prediction == 1 else "Lower risk indicated"
    })


if __name__ == "__main__":
    app.run(debug=False, port=5000)
