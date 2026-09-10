"""
WEEK 6 INTEGRATIVE CAPSTONE PROJECT
Titanic Passenger Survival Data Science Pipeline

Phases:
1. Data acquisition / loading
2. Cleaning and preprocessing
3. Exploratory data analysis
4. Unsupervised learning: K-Means
5. Supervised learning: Logistic Regression + Random Forest
6. Deep learning: PyTorch MLP
7. Evaluation, comparison, insights, and recommendations

Dataset:
titanic_cleaned.csv produced during Week 1.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, silhouette_score
)

# -------------------------
# 1. Load model-ready data
# -------------------------
df = pd.read_csv("titanic_cleaned.csv")

FEATURES = [
    "Pclass", "Sex_encoded", "Age", "SibSp", "Parch", "Fare",
    "FamilySize", "IsAlone", "Embarked_C", "Embarked_Q", "Embarked_S",
    "Title_Master", "Title_Miss", "Title_Mr", "Title_Mrs", "Title_Rare"
]
TARGET = "Survived"

X = df[FEATURES].astype(float)
y = df[TARGET].astype(int)

# -------------------------
# 2. EDA
# -------------------------
print("Shape:", df.shape)
print("Survival rate:", y.mean())
print("\nSurvival by sex:")
print(df.groupby("Sex_encoded")[TARGET].mean())
print("\nSurvival by class:")
print(df.groupby("Pclass")[TARGET].mean())

# -------------------------
# 3. Unsupervised learning
# -------------------------
cluster_features = ["Age", "Fare", "FamilySize"]
cluster_scaler = StandardScaler()
X_cluster = cluster_scaler.fit_transform(df[cluster_features])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_cluster)

print("\nCluster counts:")
print(df["Cluster"].value_counts().sort_index())
print("Silhouette score:", silhouette_score(X_cluster, df["Cluster"]))

# -------------------------
# 4. Supervised learning
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scale = StandardScaler()
X_train_scaled = scale.fit_transform(X_train)
X_test_scaled = scale.transform(X_test)

logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train_scaled, y_train)
lr_prob = logreg.predict_proba(X_test_scaled)[:,1]
lr_pred = (lr_prob >= 0.5).astype(int)

rf = RandomForestClassifier(
    n_estimators=300, max_depth=None, min_samples_leaf=2,
    random_state=42
)
rf.fit(X_train, y_train)
rf_prob = rf.predict_proba(X_test)[:,1]
rf_pred = (rf_prob >= 0.5).astype(int)

def evaluate(name, actual, pred, prob):
    print(f"\n{name}")
    print("Accuracy :", accuracy_score(actual, pred))
    print("Precision:", precision_score(actual, pred))
    print("Recall   :", recall_score(actual, pred))
    print("F1       :", f1_score(actual, pred))
    print("ROC-AUC  :", roc_auc_score(actual, prob))

evaluate("Logistic Regression", y_test, lr_pred, lr_prob)
evaluate("Random Forest", y_test, rf_pred, rf_prob)

# -------------------------
# 5. Deep learning
# -------------------------
# The complete PyTorch implementation is in deep_learning_titanic.py
# from the Week 5 submission package. It uses:
# 16 inputs -> Dense(32) -> ReLU -> Dropout -> Dense(16)
# -> ReLU -> Dropout -> output logit -> sigmoid probability.
#
# Week 5 held-out results:
# Accuracy 0.8380, Precision 0.8704, Recall 0.6812,
# F1 0.7642, ROC-AUC 0.8704.

# -------------------------
# 6. Final interpretation
# -------------------------
print("\nCAPSTONE SUMMARY")
print("- The cleaned Titanic data contains 891 rows and 20 columns.")
print("- EDA identifies sex and passenger class as strong survival-related factors.")
print("- K-Means segments passengers using Age, Fare and FamilySize.")
print("- Logistic Regression is the strongest overall Week 4/5 benchmark.")
print("- The neural network provides a deep-learning extension but does not automatically outperform simpler models.")
