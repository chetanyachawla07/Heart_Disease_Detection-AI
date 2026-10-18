"""Streamlit demo.  Run with:  streamlit run app.py"""
import pandas as pd
import streamlit as st

from heart_disease.predictor import Predictor, ValidationError
from heart_disease.schema import FEATURES

st.set_page_config(page_title="Heart Disease Risk Predictor", layout="centered")


@st.cache_resource
def get_predictor() -> Predictor:
    return Predictor.load()


st.title("❤️ Heart Disease Risk Predictor")
st.caption("Educational demo - not a medical device. Do not use for real diagnosis.")

try:
    predictor = get_predictor()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

info = predictor.summary()
st.caption(
    f"{info['model_name']} trained on {info['n_patients']} patients - "
    f"cross-validated ROC-AUC {info['roc_auc']:.2f} ± {info['roc_auc_std']:.2f}"
)

values = {}
with st.form("patient_form"):
    cols = st.columns(2)
    for i, f in enumerate(FEATURES):
        with cols[i % 2]:
            label = f"{f.label} ({f.unit})" if f.unit else f.label
            if f.kind == "choice":
                codes = list(f.choice_codes)
                names = dict(f.choices)
                values[f.name] = st.selectbox(
                    label, codes, index=codes.index(int(f.default)),
                    format_func=lambda c, names=names: names[c], help=f.help or None,
                )
            elif f.integer:
                values[f.name] = st.number_input(
                    label, int(f.min), int(f.max), int(f.default), step=int(f.step),
                    help=f.help or None)
            else:
                values[f.name] = st.number_input(
                    label, float(f.min), float(f.max), float(f.default), step=float(f.step),
                    help=f.help or None)
    submitted = st.form_submit_button("Assess risk")

if submitted:
    try:
        result = predictor.predict(values)
    except ValidationError as exc:
        for field, msg in exc.errors.items():
            st.error(f"{field}: {msg}")
        st.stop()

    st.divider()
    message = f"{result.label} - estimated probability {result.probability:.1%}"
    (st.error if result.flagged else st.success)(message)
    st.progress(min(max(result.probability, 0.0), 1.0))
    st.caption(f"The flag is raised at {result.threshold:.0%}, chosen to catch most real cases.")
    for w in result.warnings:
        st.warning(w)
    if result.factors:
        st.subheader("What moved the estimate")
        st.caption("Compared with a typical patient in the training data "
                   "(odds ratio > 1 raises risk, < 1 lowers it).")
        st.dataframe(
            pd.DataFrame([
                {"Factor": f.label, "Value": f.value,
                 "Effect": f.to_dict()["direction"], "Odds ×": round(f.odds_ratio, 2)}
                for f in result.factors
            ]),
            hide_index=True, use_container_width=True,
        )
    st.caption("Trained on ~300 records. Not clinically validated.")
