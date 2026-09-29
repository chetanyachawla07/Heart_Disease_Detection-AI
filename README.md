# Heart Disease Detection — Starter Project
BY Chetanya Aniket Tatsal Arnav
A beginner-friendly machine learning project that predicts whether a patient
has heart disease based on 13 clinical measurements (age, cholesterol, blood
pressure, etc.). Built with Python + scikit-learn.

**⚠️ Educational project only.** This is not a validated medical device and
should never be used for real diagnosis.

## Dataset

The UCI Heart Disease dataset (303 patient records, 13 features +
1 label). Already included at `data/heart.csv`.

| Column | Meaning |
|---|---|
| age | Age in years |
| sex | 1 = male, 0 = female |
| cp | Chest pain type (0–3) |
| trestbps | Resting blood pressure (mm Hg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl (1 = true) |
| restecg | Resting ECG results (0–2) |
| thalach | Max heart rate achieved |
| exang | Exercise-induced angina (1 = yes) |
| oldpeak | ST depression induced by exercise |
| slope | Slope of the peak exercise ST segment |
| ca | Number of major vessels colored by fluoroscopy (0–3) |
| thal | Thalassemia (0–3) |
| target | **Label**: 1 = disease present, 0 = no disease |

## Project structure

```
heart_disease_project/
├── data/
│   └── heart.csv              # dataset
├── models/                    # saved model + scaler (created by train.py)
├── outputs/                   # EDA plots, confusion matrix, ROC curve, comparison table
├── train.py                   # main pipeline: load → explore → preprocess → train → evaluate → save
├── predict.py                 # load saved model, predict on one new patient
├── app.py                     # interactive Streamlit demo
├── requirements.txt
└── README.md
```

## How to run it

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the models**
   ```bash
   python train.py
   ```
   This will:
   - Load and explore the dataset
   - Save EDA charts to `outputs/`
   - Split into train/test sets, scale features
   - Train Logistic Regression, Random Forest, and Gradient Boosting
   - Pick the best model by ROC-AUC and tune it with `GridSearchCV`
   - Save evaluation charts (confusion matrix, ROC curve, feature importance) to `outputs/`
   - Save the final model to `models/heart_disease_model.pkl`

3. **Predict on a single patient**
   ```bash
   python predict.py
   ```
   Edit the `new_patient` dictionary in `predict.py` to try different values.

4. **Launch the interactive demo (Streamlit version)**
   ```bash
   streamlit run app.py
   ```
   Opens a web form where you can enter patient values and get a live prediction.

5. **Launch the full frontend (Flask + HTML/CSS/JS version)**
   ```bash
   cd webapp
   pip install flask
   python app.py
   ```
   Then open `http://localhost:5000` in your browser. This is a proper web
   page (not Streamlit) — a two-column layout with patient input fields and
   a live risk result, built with plain HTML/CSS/JS talking to a Flask API
   (`/predict`). This is the one as the "frontend."

## What the results mean

Run 1 of `train.py` on this dataset typically gets:
- **Random Forest**: ~82% accuracy, ~0.91 ROC-AUC (best of the three)
- Recall (catching actual disease cases) is prioritized over precision in
  medical screening — a false negative (missing real disease) is worse than
  a false positive (an unnecessary follow-up test).

Check `outputs/model_comparison.csv` for exact numbers on your run, and the
PNG files in `outputs/` for visuals.

## Ideas for our team to extend this

- **Try more models**: SVM, XGBoost, a small neural network (Keras/PyTorch).
- **Handle class imbalance** if you switch to a larger, less balanced dataset (SMOTE, class weights).
- **Explainability**: add SHAP values so predictions can be explained per-patient.
- **Bigger dataset**: this one has only 303 rows — look at the UCI/Kaggle
  "Heart Disease Health Indicators" dataset (~250k rows) for a more robust model.
- **Deployment**: containerize `app.py` with Docker, or turn `predict.py`
  into a Flask/FastAPI REST endpoint.

## Important things to know/present

- Only 303 records — a real clinical model needs much more data and external validation.
