"""Loading and cleaning the heart disease dataset."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import FEATURE_NAMES, INVALID_CODES

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "heart.csv"

SOURCE_TARGET = "target"
LABEL = "disease"  # 1 = heart disease present, 0 = absent (what the models predict)

# IMPORTANT: in data/heart.csv the ``target`` column is inverted relative to its
# documentation: target == 1 marks patients WITHOUT disease. Evidence (see
# tests/test_data.py and README "Data notes"): the target == 1 group is younger,
# has a higher max heart rate, lower ST depression, far less exercise angina and
# fewer blocked vessels. The original project treated target as "1 = disease",
# which made every prediction backwards. We flip it once, here.
SOURCE_TARGET_IS_INVERTED = True


@dataclass
class CleaningReport:
    rows_raw: int
    duplicates_dropped: int
    invalid_codes_set_missing: dict[str, int]
    rows_clean: int
    positives: int

    def to_dict(self) -> dict:
        return asdict(self)


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    # utf-8-sig strips the byte-order mark that otherwise ends up in the first
    # column name ("\ufeffage").
    return pd.read_csv(path or DEFAULT_DATA_PATH, encoding="utf-8-sig")


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Validate columns, drop duplicates, null invalid codes, build the label."""
    missing = [c for c in [*FEATURE_NAMES, SOURCE_TARGET] if c not in raw.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df = raw[[*FEATURE_NAMES, SOURCE_TARGET]].copy()
    df = df.apply(pd.to_numeric, errors="raise")

    n_raw = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_dupes = n_raw - len(df)

    nulled: dict[str, int] = {}
    for col, bad in INVALID_CODES.items():
        mask = df[col].isin(bad)
        nulled[col] = int(mask.sum())
        df[col] = df[col].astype(float).mask(mask, np.nan)

    src = df.pop(SOURCE_TARGET)
    df[LABEL] = (1 - src) if SOURCE_TARGET_IS_INVERTED else src
    df[LABEL] = df[LABEL].astype(int)

    report = CleaningReport(
        rows_raw=n_raw,
        duplicates_dropped=n_dupes,
        invalid_codes_set_missing=nulled,
        rows_clean=len(df),
        positives=int(df[LABEL].sum()),
    )
    return df, report


def load_dataset(
    path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.Series, CleaningReport]:
    df, report = clean(load_raw(path))
    return df[FEATURE_NAMES], df[LABEL], report


def file_sha256(path: str | Path | None = None) -> str:
    return hashlib.sha256(Path(path or DEFAULT_DATA_PATH).read_bytes()).hexdigest()
