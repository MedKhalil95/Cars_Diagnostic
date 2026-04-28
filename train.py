"""
train.py — trains CarDiagnosticNet on the extended dataset that includes 'model'.
"""

import pandas as pd
import torch
from torch.utils.data import DataLoader
from network import CarDiagnosticNet
from dataset import CarDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import numpy as np

# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv("../data/processed/train_data.csv")

# ── Encode categoricals ───────────────────────────────────────────────────────
brand_enc  = LabelEncoder()
model_enc  = LabelEncoder()
engine_enc = LabelEncoder()
health_enc = LabelEncoder()

df["brand"]         = brand_enc.fit_transform(df["brand"])
df["model"]         = model_enc.fit_transform(df["model"])
df["engine_type"]   = engine_enc.fit_transform(df["engine_type"])
df["engine_health"] = health_enc.fit_transform(df["engine_health"])

# ── Features / target ────────────────────────────────────────────────────────
feature_cols = [c for c in df.columns if c != "engine_health"]
X = df[feature_cols].values.astype(np.float32)
y = df["engine_health"].values.astype(np.int64)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Datasets & loaders ───────────────────────────────────────────────────────
train_ds = CarDataset(X_train, y_train)
test_ds  = CarDataset(X_test,  y_test)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False)

# ── Model ────────────────────────────────────────────────────────────────────
num_classes = len(health_enc.classes_)
model     = CarDiagnosticNet(input_size=X.shape[1], num_classes=num_classes)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# ── Training loop ─────────────────────────────────────────────────────────────
EPOCHS = 30
for epoch in range(EPOCHS):
    model.train()
    total_loss, n_batches = 0.0, 0

    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(x_batch), y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches  += 1

    scheduler.step()
    print(f"Epoch {epoch+1:02d}/{EPOCHS} | Loss {total_loss/n_batches:.4f}")

# ── Evaluation ───────────────────────────────────────────────────────────────
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        preds = torch.argmax(model(x_batch), dim=1)
        all_preds.extend(preds.tolist())
        all_labels.extend(y_batch.tolist())

print("\n── Test Set Report ──")
print(classification_report(all_labels, all_preds, target_names=health_enc.classes_))

# ── Persist artefacts ────────────────────────────────────────────────────────
torch.save(model.state_dict(), "model.pt")
joblib.dump(brand_enc,    "brand_encoder.pkl")
joblib.dump(model_enc,    "model_encoder.pkl")
joblib.dump(engine_enc,   "engine_encoder.pkl")
joblib.dump(health_enc,   "health_encoder.pkl")
joblib.dump(feature_cols, "feature_cols.pkl")
print("\nSaved: model.pt, *_encoder.pkl, feature_cols.pkl")