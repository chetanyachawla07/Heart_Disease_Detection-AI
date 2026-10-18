# Model card: Heart Disease Risk Estimator

**Status:** educational prototype. **Not a medical device. Not for clinical use.**

## Intended use
Teaching and demonstration of an end-to-end ML workflow (cleaning, honest evaluation, serving,
explanation). Not intended to screen, diagnose or advise real patients.

## Model
- Chosen automatically by `python train.py`; currently **logistic regression** (L2, `C = 0.1`).
- Pipeline: median imputation + standardisation for numeric inputs; most-frequent imputation for
  binary inputs; one-hot encoding for `cp`, `restecg`, `slope`, `thal`.
- Output: probability of heart disease and a binary flag raised at **0.27**.
- The saved artifact (`models/heart_model.joblib`) bundles the pipeline, threshold, training
  ranges, reference values for explanations, metrics, and the scikit-learn version used.

## Data
- 303 rows (302 after dropping 1 duplicate) of the Cleveland-style UCI heart disease data, 138 with
  disease (46%).
- The raw `target` column is inverted relative to its documentation; the pipeline corrects this.
  `ca = 4` and `thal = 0` are invalid codes treated as missing. Details: README "Data notes".
- One source, one era, one population. Demographic composition is not documented in the file.

## Evaluation
Nested repeated stratified cross-validation (5 outer folds × 2 repeats, 5-fold inner tuning),
mean ± std over outer folds:

| Metric | Value |
|---|---|
| ROC-AUC | 0.915 ± 0.050 |
| Accuracy @ 0.5 | 0.847 |
| Cohen's Kappa @ 0.5 | 0.689 |
| MCC @ 0.5 | 0.697 |
| Brier score | 0.117 (baseline 0.248) |

At the chosen threshold (0.27), on out-of-fold predictions: recall 0.91, specificity 0.71,
precision 0.72. The threshold was tuned on those same predictions, so these are mildly optimistic.
Calibration (`outputs/06_calibration.png`) is close to the diagonal.

No external validation set exists. Subgroup performance (sex, age) has not been analysed.

## Limitations and risks
- **Tiny, single-source data**: wide uncertainty (AUC std ≈ 0.05 across folds); likely to degrade on
  other populations.
- **Missing context**: no medications, history, symptoms timeline or family history.
- **Associational explanations**: the "what moved the estimate" panel is a one-at-a-time what-if and
  does not imply causation.
- **Inputs outside the training range** (e.g. age > 77) trigger a warning; predictions there are
  extrapolations.
- **Coding ambiguity**: the meaning of the chest-pain codes (`cp`) could not be verified from the
  data file, so they are shown as neutral "Type 0-3".
- **Automation bias**: a confident-looking percentage can be over-trusted. The UI keeps the
  disclaimer visible and avoids diagnostic language ("higher risk indicated", not "you have…").

## Reproduce
```bash
pip install -r requirements.txt
python train.py --seed 42
```
`outputs/metrics.json` records the seed, data hash, library versions and every number above.
