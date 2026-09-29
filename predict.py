"""
Load the trained model and make a prediction for a single new patient.
Run with:  python predict.py
"""
import joblib
import pandas as pd

model = joblib.load("models/heart_disease_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_names = joblib.load("models/feature_names.pkl")

# Example patient - replace these values with a real record to test
new_patient = {
    "age": 58, "sex": 1, "cp": 0, "trestbps": 140, "chol": 289,
    "fbs": 0, "restecg": 0, "thalach": 145, "exang": 1,
    "oldpeak": 0.8, "slope": 1, "ca": 1, "thal": 3
}

X_new = pd.DataFrame([new_patient])[feature_names]
X_new_scaled = scaler.transform(X_new)

prediction = model.predict(X_new_scaled)[0]
probability = model.predict_proba(X_new_scaled)[0][1]

print("Patient data:", new_patient)
print(f"\nPrediction: {'Heart Disease Likely' if prediction == 1 else 'No Heart Disease Detected'}")
print(f"Model confidence (probability of disease): {probability:.1%}")
