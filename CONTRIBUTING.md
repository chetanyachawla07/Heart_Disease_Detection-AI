# Contributing

Thanks for helping improve the project! This is an educational repo, so clarity beats cleverness.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

## Before you open a pull request
```bash
ruff check .        # lint (ruff check . --fix for auto-fixes)
pytest              # ~10 s; includes a fast end-to-end training run
```
CI runs the same two commands on Python 3.10-3.12.

## Workflow
1. Branch from `main`: `git checkout -b feat/short-description`
2. Keep PRs focused: one idea per PR, with tests for new behaviour.
3. Describe *what changed and why* in the PR, and attach before/after numbers if you touched modelling.

## Where things live
- **Add or change a feature** (label, valid range, codes) → `heart_disease/schema.py` only. The
  forms, API validation and CLI are generated from it.
- **Add a model** → `heart_disease/modeling.py::candidates` (and give it a rank in `COMPLEXITY`).
- **Change cleaning** → `heart_disease/data.py`, and update `tests/test_data.py`.

## Retraining and committed artifacts
`models/` and `outputs/` are committed so the apps work out of the box. If your change affects
modelling or data, run `python train.py` and commit the regenerated files, and put the new metrics
in the PR description. Don't hand-edit them. Pin to the same scikit-learn minor version as the
Dockerfile when regenerating `models/heart_model.joblib`.

## Safety rules
- Never claim or imply clinical validity. Keep the disclaimers.
- Don't add patient-identifying data to the repo.
- Model files are pickles: don't commit models from untrusted sources.
