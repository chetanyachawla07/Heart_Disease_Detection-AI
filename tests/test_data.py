import numpy as np
import pandas as pd
import pytest

from heart_disease.data import LABEL, clean, load_dataset, load_raw
from heart_disease.schema import FEATURE_NAMES


def test_first_column_has_no_bom():
    assert list(load_raw().columns)[0] == "age"


def test_label_direction_is_clinically_sensible():
    """Regression test for the inverted-label bug.

    The raw ``target`` column marks HEALTHY patients with 1. If someone removes
    the flip, these relationships reverse and this test fails.
    """
    X, y, _ = load_dataset()
    sick, well = X[y == 1], X[y == 0]
    assert sick["oldpeak"].mean() > well["oldpeak"].mean()      # more ST depression
    assert sick["thalach"].mean() < well["thalach"].mean()      # lower max heart rate
    assert sick["exang"].mean() > well["exang"].mean()          # more exercise angina
    assert sick["ca"].mean() > well["ca"].mean()                # more blocked vessels


def test_shipped_dataset_is_cleaned():
    X, y, report = load_dataset()
    assert list(X.columns) == FEATURE_NAMES
    assert len(X) == len(y) == report.rows_clean
    assert not X.duplicated().any()
    assert not (X["ca"] == 4).any()
    assert not (X["thal"] == 0).any()
    assert set(y.unique()) == {0, 1}


def test_clean_on_synthetic_frame():
    base = dict(age=50, sex=1, cp=0, trestbps=120, chol=200, fbs=0, restecg=0, thalach=150,
                exang=0, oldpeak=1.0, slope=1, ca=0, thal=2, target=1)
    raw = pd.DataFrame([
        base,
        base,                                   # exact duplicate
        {**base, "age": 60, "ca": 4},           # invalid ca
        {**base, "age": 61, "thal": 0, "target": 0},  # invalid thal, other class
    ])
    df, report = clean(raw)
    assert report.duplicates_dropped == 1
    assert report.invalid_codes_set_missing == {"ca": 1, "thal": 1}
    assert len(df) == 3
    assert np.isnan(df.loc[df.age == 60, "ca"]).all()
    assert np.isnan(df.loc[df.age == 61, "thal"]).all()
    # target is inverted: source 1 -> disease 0, source 0 -> disease 1
    assert df.loc[df.age == 50, LABEL].iloc[0] == 0
    assert df.loc[df.age == 61, LABEL].iloc[0] == 1


def test_missing_column_is_reported():
    raw = load_raw().drop(columns=["chol"])
    with pytest.raises(ValueError, match="chol"):
        clean(raw)
