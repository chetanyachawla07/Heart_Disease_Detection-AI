import pytest

from heart_disease.predictor import ValidationError, validate_patient
from heart_disease.schema import EXAMPLE_PATIENT, FEATURE_NAMES


def errors_for(**override):
    with pytest.raises(ValidationError) as exc:
        validate_patient({**EXAMPLE_PATIENT, **override})
    return exc.value.errors


def test_example_patient_is_valid():
    clean = validate_patient(EXAMPLE_PATIENT)
    assert set(clean) == set(FEATURE_NAMES)


def test_numeric_strings_are_accepted():
    clean = validate_patient({**EXAMPLE_PATIENT, "age": "58", "oldpeak": "0.8"})
    assert clean["age"] == 58 and clean["oldpeak"] == 0.8


def test_extra_keys_are_ignored():
    assert "hacker" not in validate_patient({**EXAMPLE_PATIENT, "hacker": 1})


def test_missing_and_blank_fields():
    patient = dict(EXAMPLE_PATIENT)
    del patient["chol"]
    patient["age"] = ""
    with pytest.raises(ValidationError) as exc:
        validate_patient(patient)
    assert set(exc.value.errors) == {"chol", "age"}


def test_out_of_range_numbers():
    errs = errors_for(age=500, trestbps=10, oldpeak=99)
    assert set(errs) == {"age", "trestbps", "oldpeak"}


def test_invalid_choice_codes():
    # ca=4 and thal=0 are exactly the invalid codes found in the raw CSV
    assert set(errors_for(cp=9, ca=4, thal=0, sex=2)) == {"cp", "ca", "thal", "sex"}


def test_non_numeric_and_non_finite():
    errs = errors_for(age="abc", chol="nan", thalach=float("inf"))
    assert set(errs) == {"age", "chol", "thalach"}


def test_fractional_values_rejected_where_integer_required():
    assert "age" in errors_for(age=55.5)
    assert "cp" in errors_for(cp=1.5)
    assert validate_patient({**EXAMPLE_PATIENT, "oldpeak": 1.5})["oldpeak"] == 1.5
