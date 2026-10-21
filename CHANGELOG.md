# Changelog

## 2.1.0

### Added
- Gaussian Naive Bayes as a sixth candidate model (from the course Experiment 1.3 notebook).
- Cohen's Kappa and Matthews correlation coefficient (MCC) in the cross-validated metrics,
  `metrics.json` and `model_comparison.csv`.

## 2.0.0

### Fixed
- **Predictions were inverted.** The CSV's `target` marks healthy patients with 1; v1 treated it
  as "disease". Corrected in `heart_disease/data.py` with regression tests.
- Wrong `thal` labels in the web form (1 was shown as "Normal"; it is "Fixed defect", 2 is Normal).
- Invalid dataset codes (`ca=4`, `thal=0`) and one duplicate row are now handled.
- Model selection used the test set it was scored on; replaced with nested cross-validation.
- Web app crashed on missing/invalid JSON; now returns structured `400` errors.
- Relative model paths broke when launched from another directory.
- Typo "Protype" on the web page.

### Added
- `heart_disease` package (schema, data, modeling, training, predictor, reporting).
- Leakage-free sklearn `Pipeline`; SVM added; majority baseline for context.
- One-standard-error model selection; recall-targeted decision threshold.
- Per-patient explanations ("what moved the estimate") and out-of-range warnings.
- Input validation, `/health` and `/api/schema` endpoints, security headers, risk meter in the UI.
- 36 tests, GitHub Actions CI, Dockerfile, MODEL_CARD, CONTRIBUTING, `pyproject.toml`.
- `python predict.py --age ... --json` CLI.

### Changed
- Single model artifact `models/heart_model.joblib` replaces six pickle files (the web app's
  copy was a duplicate).
- Web form and Streamlit form are generated from one schema.
- `requirements.txt` now includes Flask; Streamlit moved to `requirements-streamlit.txt`.
- Output plots renamed/extended (8 figures + `metrics.json`).

### Removed
- `models/{heart_disease_model,scaler,feature_names}.pkl` and the duplicates under `webapp/models/`.
