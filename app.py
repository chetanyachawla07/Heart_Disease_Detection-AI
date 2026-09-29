"""
Interactive demo app for the Heart Disease Detection model.
Run with:  streamlit run app.py
"""
import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Heart Disease Risk Predictor", layout="centered")

model = joblib.load("models/heart_disease_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_names = joblib.load("models/feature_names.pkl")

st.title("❤️ Heart Disease Risk Predictor")
st.caption("Educational demo — not a medical device. Do not use for real diagnosis.")

with st.form("patient_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 20, 100, 55)
        sex = st.selectbox("Sex", ["Female", "Male"])
        cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 220, 130)
        chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 240)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
        restecg = st.selectbox("Resting ECG Results (0-2)", [0, 1, 2])
    with col2:
        thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
        exang = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
        oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 10.0, 1.0, step=0.1)
        slope = st.selectbox("Slope of Peak Exercise ST Segment (0-2)", [0, 1, 2])
        ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
        thal = st.selectbox("Thalassemia (0-3)", [0, 1, 2, 3])

    submitted = st.form_submit_button("Predict")

if submitted:
    patient = {
        "age": age, "sex": 1 if sex == "Male" else 0, "cp": cp,
        "trestbps": trestbps, "chol": chol,
        "fbs": 1 if fbs == "Yes" else 0, "restecg": restecg,
        "thalach": thalach, "exang": 1 if exang == "Yes" else 0,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }
    X_new = pd.DataFrame([patient])[feature_names]
    X_scaled = scaler.transform(X_new)
    pred = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0][1]

    st.divider()
    if pred == 1:
        st.error(f"⚠️ Higher risk indicated — estimated probability: {prob:.1%}")
    else:
        st.success(f"✅ Lower risk indicated — estimated probability: {prob:.1%}")
    st.caption("This is a demo model trained on 303 records. It is not clinically validated.")
