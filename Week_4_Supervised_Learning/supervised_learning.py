"""
Week 4 Task: Supervised Learning Model Implementation
Problem Type: Classification (Predict Passenger Survival)
Dataset: Titanic Cleaned Dataset (from Week 1 preprocessing)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110
RANDOM_STATE = 42

# ---------------------------------------------------------------
# STEP 1: LOAD DATA
# ---------------------------------------------------------------
df = pd.read_csv("titanic_cleaned.csv")
print("=== STEP 1: DATA LOADING ===")
print(f"Shape: {df.shape}")
print(df.head())

# ---------------------------------------------------------------
# STEP 2: PROBLEM DEFINITION
# ---------------------------------------------------------------
# Classification problem: predict 'Survived' (0 = No, 1 = Yes)
# using demographic, family, fare and embarkation features.

# ---------------------------------------------------------------
# STEP 3: FEATURE ENGINEERING / SELECTION
# ---------------------------------------------------------------
print("\n=== STEP 3: FEATURE SELECTION ===")

feature_cols = [
    "Pclass", "Sex_encoded", "Age", "SibSp", "Parch", "Fare",
    "FamilySize", "IsAlone",
    "Embarked_C", "Embarked_Q", "Embarked_S",
    "Title_Master", "Title_Miss", "Title_Mr", "Title_Mrs", "Title_Rare",
]
target_col = "Survived"

X = df[feature_cols].copy()
# Ensure boolean dummy columns are numeric (0/1)
for col in X.columns:
    if X[col].dtype == bool:
        X[col] = X[col].astype(int)
y = df[target_col].copy()

print(f"Features used ({len(feature_cols)}): {feature_cols}")
print(f"Target: {target_col}")
print(f"Class balance:\n{y.value_counts(normalize=True).round(3)}")

# ---------------------------------------------------------------
# STEP 4: TRAIN / TEST SPLIT
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")

# Scale features for Logistic Regression (tree model does not need scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------
# STEP 5: MODEL 1 - LOGISTIC REGRESSION (baseline, interpretable)
# ---------------------------------------------------------------
print("\n=== STEP 5: MODEL 1 - LOGISTIC REGRESSION ===")
log_reg = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
log_reg.fit(X_train_scaled, y_train)
y_pred_lr = log_reg.predict(X_test_scaled)
y_proba_lr = log_reg.predict_proba(X_test_scaled)[:, 1]

lr_metrics = {
    "Accuracy": accuracy_score(y_test, y_pred_lr),
    "Precision": precision_score(y_test, y_pred_lr),
    "Recall": recall_score(y_test, y_pred_lr),
    "F1-Score": f1_score(y_test, y_pred_lr),
    "ROC-AUC": roc_auc_score(y_test, y_proba_lr),
}
print("Logistic Regression Test Metrics:")
for k, v in lr_metrics.items():
    print(f"  {k}: {v:.4f}")

# 5-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
lr_cv_scores = cross_val_score(log_reg, X_train_scaled, y_train, cv=cv, scoring="accuracy")
print(f"5-Fold CV Accuracy: {lr_cv_scores.mean():.4f} (+/- {lr_cv_scores.std():.4f})")

# ---------------------------------------------------------------
# STEP 6: MODEL 2 - RANDOM FOREST (non-linear, ensemble)
# ---------------------------------------------------------------
print("\n=== STEP 6: MODEL 2 - RANDOM FOREST CLASSIFIER ===")
rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=RANDOM_STATE)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

rf_metrics = {
    "Accuracy": accuracy_score(y_test, y_pred_rf),
    "Precision": precision_score(y_test, y_pred_rf),
    "Recall": recall_score(y_test, y_pred_rf),
    "F1-Score": f1_score(y_test, y_pred_rf),
    "ROC-AUC": roc_auc_score(y_test, y_proba_rf),
}
print("Random Forest Test Metrics:")
for k, v in rf_metrics.items():
    print(f"  {k}: {v:.4f}")

rf_cv_scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring="accuracy")
print(f"5-Fold CV Accuracy: {rf_cv_scores.mean():.4f} (+/- {rf_cv_scores.std():.4f})")

# ---------------------------------------------------------------
# STEP 7: HYPERPARAMETER TUNING (Random Forest via GridSearchCV)
# ---------------------------------------------------------------
print("\n=== STEP 7: HYPERPARAMETER TUNING (Random Forest) ===")
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [4, 6, 8, None],
    "min_samples_leaf": [1, 2, 4],
}
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=RANDOM_STATE),
    param_grid, cv=cv, scoring="accuracy", n_jobs=-1
)
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best CV Accuracy: {grid_search.best_score_:.4f}")

y_pred_best = best_rf.predict(X_test)
y_proba_best = best_rf.predict_proba(X_test)[:, 1]
best_metrics = {
    "Accuracy": accuracy_score(y_test, y_pred_best),
    "Precision": precision_score(y_test, y_pred_best),
    "Recall": recall_score(y_test, y_pred_best),
    "F1-Score": f1_score(y_test, y_pred_best),
    "ROC-AUC": roc_auc_score(y_test, y_proba_best),
}
print("Tuned Random Forest Test Metrics:")
for k, v in best_metrics.items():
    print(f"  {k}: {v:.4f}")

# ---------------------------------------------------------------
# STEP 8: MODEL COMPARISON TABLE
# ---------------------------------------------------------------
comparison = pd.DataFrame(
    [lr_metrics, rf_metrics, best_metrics],
    index=["Logistic Regression", "Random Forest (default)", "Random Forest (tuned)"]
).round(4)
print("\n=== STEP 8: MODEL COMPARISON ===")
print(comparison)
comparison.to_csv("model_comparison.csv")

# ---------------------------------------------------------------
# STEP 9: VISUALIZATIONS
# ---------------------------------------------------------------
print("\n=== STEP 9: GENERATING VISUALIZATIONS ===")

# Confusion Matrix - Best Model
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Did Not Survive", "Survived"],
            yticklabels=["Did Not Survive", "Survived"])
plt.title("Confusion Matrix - Tuned Random Forest")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

# ROC Curves comparison
plt.figure(figsize=(6, 5))
for name, proba in [("Logistic Regression", y_proba_lr),
                     ("Random Forest (default)", y_proba_rf),
                     ("Random Forest (tuned)", y_proba_best)]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc_val = roc_auc_score(y_test, proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.close()

# Model comparison bar chart
plt.figure(figsize=(7, 4))
comparison.plot(kind="bar", figsize=(8, 5))
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=15)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("model_comparison.png")
plt.close()

# Feature importance - tuned Random Forest
importances = pd.Series(best_rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
plt.figure(figsize=(7, 5))
sns.barplot(x=importances.values, y=importances.index, palette="viridis")
plt.title("Feature Importance - Tuned Random Forest")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()
print("Top 5 features:")
print(importances.head(5))

# Logistic Regression coefficients
coef = pd.Series(log_reg.coef_[0], index=feature_cols).sort_values()
plt.figure(figsize=(7, 5))
coef.plot(kind="barh", color="#5b8c85")
plt.title("Logistic Regression Coefficients")
plt.xlabel("Coefficient Value (standardized features)")
plt.tight_layout()
plt.savefig("logreg_coefficients.png")
plt.close()

# ---------------------------------------------------------------
# STEP 10: CLASSIFICATION REPORTS
# ---------------------------------------------------------------
print("\n=== STEP 10: DETAILED CLASSIFICATION REPORTS ===")
print("\nLogistic Regression:\n", classification_report(y_test, y_pred_lr))
print("\nTuned Random Forest:\n", classification_report(y_test, y_pred_best))

print("\n=== WEEK 4 SUPERVISED LEARNING TASK COMPLETE ===")
