"""Single source of truth for the 13 clinical features.

The training code, the validation layer, the Flask form, the Streamlit form
and the CLI are all generated from ``FEATURES`` so labels, codes and valid
ranges can never drift apart between them.

Coding notes (checked against the data in ``data/heart.csv``):

* ``thal``: 1 = fixed defect, 2 = normal, 3 = reversible defect. The old UI
  labelled 1 as "Normal", which contradicts the data (thal=2 is the healthiest
  group). 0 is not a valid code and is treated as missing.
* ``ca``: valid values are 0-3. 4 is not valid and is treated as missing.
* ``cp``: the dataset's documented order (typical / atypical / non-anginal /
  asymptomatic) does not match how the values behave in the data, so the UI
  shows neutral "Type 0-3" labels instead of guessing. See README, "Data notes".
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Feature:
    name: str
    label: str
    kind: str  # "number" (free numeric input) or "choice" (fixed codes)
    role: str  # modelling role: "numeric", "binary" or "nominal"
    default: float
    unit: str = ""
    min: float | None = None  # hard validation limits for "number" features
    max: float | None = None
    step: float = 1.0
    integer: bool = True
    choices: tuple[tuple[int, str], ...] = ()
    help: str = ""

    @property
    def choice_codes(self) -> tuple[int, ...]:
        return tuple(code for code, _ in self.choices)

    def display_value(self, value: float) -> str:
        """Human-readable rendering of a value, e.g. 'Male' or '140 mm Hg'."""
        if self.kind == "choice":
            for code, text in self.choices:
                if code == int(value):
                    return text
            return str(value)
        text = f"{value:g}"
        return f"{text} {self.unit}".strip()

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "kind": self.kind,
            "default": self.default,
            "unit": self.unit,
            "min": self.min,
            "max": self.max,
            "step": self.step,
            "choices": [{"value": c, "label": t} for c, t in self.choices],
            "help": self.help,
        }


_YES_NO = ((0, "No"), (1, "Yes"))

FEATURES: tuple[Feature, ...] = (
    Feature("age", "Age", "number", "numeric", 55, unit="years", min=18, max=100),
    Feature("sex", "Sex", "choice", "binary", 1, choices=((0, "Female"), (1, "Male"))),
    Feature(
        "cp", "Chest pain type", "choice", "nominal", 0,
        choices=((0, "Type 0"), (1, "Type 1"), (2, "Type 2"), (3, "Type 3")),
        help="Coded 0-3 as in the dataset. The dataset's own descriptions of "
             "these codes are unreliable, so no clinical names are shown.",
    ),
    Feature("trestbps", "Resting blood pressure", "number", "numeric", 130,
            unit="mm Hg", min=70, max=250),
    Feature("chol", "Serum cholesterol", "number", "numeric", 240,
            unit="mg/dl", min=100, max=600),
    Feature("fbs", "Fasting blood sugar > 120 mg/dl", "choice", "binary", 0, choices=_YES_NO),
    Feature(
        "restecg", "Resting ECG result", "choice", "nominal", 0,
        choices=((0, "Normal"), (1, "ST-T wave abnormality"),
                 (2, "Left ventricular hypertrophy")),
    ),
    Feature("thalach", "Maximum heart rate achieved", "number", "numeric", 150,
            unit="bpm", min=60, max=230),
    Feature("exang", "Exercise-induced angina", "choice", "binary", 0, choices=_YES_NO),
    Feature("oldpeak", "ST depression (oldpeak)", "number", "numeric", 1.0,
            unit="mm", min=0, max=10, step=0.1, integer=False,
            help="ST depression induced by exercise relative to rest."),
    Feature(
        "slope", "Slope of peak exercise ST segment", "choice", "nominal", 1,
        choices=((0, "Downsloping"), (1, "Flat"), (2, "Upsloping")),
    ),
    Feature(
        "ca", "Major vessels coloured by fluoroscopy", "choice", "numeric", 0,
        choices=((0, "0"), (1, "1"), (2, "2"), (3, "3")),
    ),
    Feature(
        "thal", "Thalassemia", "choice", "nominal", 2,
        choices=((1, "Fixed defect"), (2, "Normal"), (3, "Reversible defect")),
    ),
)

FEATURE_NAMES: list[str] = [f.name for f in FEATURES]
FEATURES_BY_NAME: dict[str, Feature] = {f.name: f for f in FEATURES}
NUMERIC_COLS = [f.name for f in FEATURES if f.role == "numeric"]
BINARY_COLS = [f.name for f in FEATURES if f.role == "binary"]
NOMINAL_COLS = [f.name for f in FEATURES if f.role == "nominal"]

# Codes that appear in the raw CSV but are not valid values; treated as missing.
INVALID_CODES: dict[str, set[int]] = {"ca": {4}, "thal": {0}}

# A realistic example used by the CLI (predict.py) and docs.
EXAMPLE_PATIENT: dict[str, float] = {
    "age": 58, "sex": 1, "cp": 0, "trestbps": 140, "chol": 289, "fbs": 0,
    "restecg": 0, "thalach": 145, "exang": 1, "oldpeak": 0.8, "slope": 1,
    "ca": 1, "thal": 3,
}
