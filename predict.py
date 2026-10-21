"""Predict for one patient from the command line.

    python predict.py                              # built-in example patient
    python predict.py --age 62 --sex 1 --chol 300  # override any field
    python predict.py --json                       # machine-readable output

Fields you don't pass fall back to the example patient.
"""
import argparse
import json
import sys

from heart_disease.predictor import Predictor, ValidationError
from heart_disease.schema import EXAMPLE_PATIENT, FEATURES


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    for f in FEATURES:
        extra = f" (codes: {', '.join(f'{c}={t}' for c, t in f.choices)})" if f.choices else ""
        ap.add_argument(f"--{f.name}", type=float, help=f"{f.label}{extra}")
    ap.add_argument("--model", help="path to a model file (default: models/heart_model.joblib)")
    ap.add_argument("--json", action="store_true", help="print JSON instead of text")
    args = ap.parse_args(argv)

    patient = {f.name: getattr(args, f.name) for f in FEATURES}
    patient = {k: (v if v is not None else EXAMPLE_PATIENT[k]) for k, v in patient.items()}

    try:
        result = Predictor.load(args.model).predict(patient)
    except (FileNotFoundError, ValidationError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps({"patient": patient, **result.to_dict()}, indent=2))
        return 0

    print("Patient:", patient)
    print(f"\n{result.label}  (estimated probability {result.probability:.1%}, "
          f"flag threshold {result.threshold:.0%})")
    for w in result.warnings:
        print(f"  ! {w}")
    if result.factors:
        print("\nWhat moved the estimate (vs. a typical patient in the training data):")
        for fac in result.factors:
            print(f"  - {fac.label} = {fac.value}: {fac.to_dict()['direction']} "
                  f"the odds of disease (x{fac.odds_ratio:.2f})")
    print("\nEducational model only - not a medical device.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
