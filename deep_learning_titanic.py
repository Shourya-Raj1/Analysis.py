"""
Week 5 Task: Deep Learning Application in Data Science
Problem: Titanic passenger survival prediction (binary classification)
Framework: PyTorch
Dataset: titanic_cleaned.csv from Week 1
Random seed: 42
"""

import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_PATH = "titanic_cleaned.csv"

FEATURES = [
    "Pclass", "Sex_encoded", "Age", "SibSp", "Parch", "Fare",
    "FamilySize", "IsAlone", "Embarked_C", "Embarked_Q", "Embarked_S",
    "Title_Master", "Title_Miss", "Title_Mr", "Title_Mrs", "Title_Rare"
]

df = pd.read_csv(DATA_PATH)
X = df[FEATURES].astype(float).values
y = df["Survived"].astype(int).values

# 80/20 stratified train/test split
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)

# 15% of training data becomes validation data
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.15,
    random_state=SEED, stratify=y_train_full
)

# Fit scaling only on the training data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

train_ds = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

val_x = torch.tensor(X_val, dtype=torch.float32)
val_y = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
test_x = torch.tensor(X_test, dtype=torch.float32)

class TitanicMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)

model = TitanicMLP()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

best_val_loss = float("inf")
best_state = None
wait = 0
patience = 25

for epoch in range(1, 301):
    model.train()

    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(val_x), val_y).item()

    if val_loss < best_val_loss - 1e-4:
        best_val_loss = val_loss
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        wait = 0
    else:
        wait += 1

    if wait >= patience:
        break

model.load_state_dict(best_state)
model.eval()

with torch.no_grad():
    probability = torch.sigmoid(model(test_x)).numpy().ravel()

prediction = (probability >= 0.5).astype(int)

print("=== WEEK 5 DEEP LEARNING RESULTS ===")
print(f"Accuracy : {accuracy_score(y_test, prediction):.4f}")
print(f"Precision: {precision_score(y_test, prediction):.4f}")
print(f"Recall   : {recall_score(y_test, prediction):.4f}")
print(f"F1-Score : {f1_score(y_test, prediction):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, probability):.4f}")
