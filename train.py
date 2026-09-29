"""
Heart Disease Detection - Training Pipeline
=============================================
A beginner-friendly, step-by-step machine learning pipeline that predicts
whether a patient has heart disease based on clinical measurements.

Dataset: UCI Cleveland Heart Disease dataset (303 patients, 13 features)

Run with:  python train.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # save plots to file instead of popping up a window
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay
)

OUT = "outputs"

# -----------------------------------------------------------------------
# STEP 1: Load the data
# -----------------------------------------------------------------------
print("STEP 1: Loading data...")
df = pd.read_csv("data/heart.csv")
print(f"  Loaded {df.shape[0]} patients, {df.shape[1]} columns")
print(f"  Columns: {list(df.columns)}")

# Column meanings (for your README / team reference):
# age      - age in years
# sex      - 1 = male, 0 = female
# cp       - chest pain type (0-3)
# trestbps - resting blood pressure (mm Hg)
# chol     - serum cholesterol (mg/dl)
# fbs      - fasting blood sugar > 120 mg/dl (1 = true)
# restecg  - resting ECG results (0-2)
# thalach  - max heart rate achieved
# exang    - exercise-induced angina (1 = yes)
# oldpeak  - ST depression induced by exercise
# slope    - slope of the peak exercise ST segment
# ca       - number of major vessels colored by fluoroscopy (0-3)
# thal     - thalassemia (0-3)
# target   - 1 = has heart disease, 0 = no heart disease  (LABEL)

# -----------------------------------------------------------------------
# STEP 2: Explore the data (EDA)
# -----------------------------------------------------------------------
print("\nSTEP 2: Exploring data...")
print(f"  Missing values per column:\n{df.isnull().sum().sum()} total missing cells")
print(f"  Target balance:\n{df['target'].value_counts()}")

# Correlation heatmap - which features relate most to heart disease?
plt.figure(figsize=(11, 9))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{OUT}/01_correlation_heatmap.png", dpi=130)
plt.close()

# Target class balance
plt.figure(figsize=(5, 4))
sns.countplot(x="target", data=df)
plt.title("Target Class Balance (0 = No Disease, 1 = Disease)")
plt.tight_layout()
plt.savefig(f"{OUT}/02_target_balance.png", dpi=130)
plt.close()

print(f"  Saved EDA plots to {OUT}/")

# -----------------------------------------------------------------------
# STEP 3: Preprocess (split features/label, scale numeric features)
# -----------------------------------------------------------------------
print("\nSTEP 3: Preprocessing...")
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------------------------------------------------
# STEP 4: Train several candidate models
# -----------------------------------------------------------------------
print("\nSTEP 4: Training models...")
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

results = []
trained_models = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="accuracy")

    results.append({
        "Model": name, "Accuracy": acc, "Precision": prec,
        "Recall": rec, "F1": f1, "ROC-AUC": auc,
        "CV Accuracy (mean)": cv_scores.mean()
    })
    trained_models[name] = model
    print(f"  {name:22s} | Acc: {acc:.3f}  Prec: {prec:.3f}  Recall: {rec:.3f}  F1: {f1:.3f}  AUC: {auc:.3f}")

results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)
results_df.to_csv(f"{OUT}/model_comparison.csv", index=False)
print(f"\n  Full comparison saved to {OUT}/model_comparison.csv")

# -----------------------------------------------------------------------
# STEP 5: Pick the best model and tune it
# -----------------------------------------------------------------------
best_name = results_df.iloc[0]["Model"]
print(f"\nSTEP 5: Best baseline model = {best_name}. Tuning hyperparameters...")

if best_name == "Random Forest":
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5],
    }
    base_model = RandomForestClassifier(random_state=42)
elif best_name == "Gradient Boosting":
    param_grid = {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [2, 3, 4],
    }
    base_model = GradientBoostingClassifier(random_state=42)
else:
    param_grid = {"C": [0.01, 0.1, 1, 10]}
    base_model = LogisticRegression(max_iter=1000)

grid = GridSearchCV(base_model, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
grid.fit(X_train_scaled, y_train)
best_model = grid.best_estimator_
print(f"  Best params: {grid.best_params_}")

# Final evaluation of tuned model
final_preds = best_model.predict(X_test_scaled)
final_probs = best_model.predict_proba(X_test_scaled)[:, 1]
print("\nFinal tuned model performance on test set:")
print(classification_report(y_test, final_preds, target_names=["No Disease", "Disease"]))

# Confusion matrix plot
cm = confusion_matrix(y_test, final_preds)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Disease", "Disease"], yticklabels=["No Disease", "Disease"])
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.title(f"Confusion Matrix - {best_name} (tuned)")
plt.tight_layout()
plt.savefig(f"{OUT}/03_confusion_matrix.png", dpi=130)
plt.close()

# ROC curve
plt.figure(figsize=(5, 5))
RocCurveDisplay.from_predictions(y_test, final_probs)
plt.title(f"ROC Curve - {best_name} (tuned)")
plt.tight_layout()
plt.savefig(f"{OUT}/04_roc_curve.png", dpi=130)
plt.close()

# Feature importance (if available)
if hasattr(best_model, "feature_importances_"):
    importance = pd.Series(best_model.feature_importances_, index=X.columns).sort_values()
    plt.figure(figsize=(7, 6))
    importance.plot(kind="barh")
    plt.title(f"Feature Importance - {best_name}")
    plt.tight_layout()
    plt.savefig(f"{OUT}/05_feature_importance.png", dpi=130)
    plt.close()

# -----------------------------------------------------------------------
# STEP 6: Save the trained model + scaler for reuse
# -----------------------------------------------------------------------
joblib.dump(best_model, "models/heart_disease_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(list(X.columns), "models/feature_names.pkl")
print("\nSTEP 6: Saved trained model to models/heart_disease_model.pkl")

print("\nDone! Check the 'outputs/' folder for plots and 'models/' for the saved model.")
