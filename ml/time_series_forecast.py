# ml/time_series_forecast.py
# TRUE TIME-SERIES PREDICTION WITH OUTLIER CONTROL
# Uses LOG TRANSFORMATION (BEST PRACTICE)

import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

from db import db
from models.sales_transaction import SalesTransaction
from models.sales_transaction_item import SalesTransactionItem
from models.item import Item


def run_time_series_forecast():
    print("\n[ML] Demand prediction started (log-scaled)")

    # ===============================
    # 1️⃣ LOAD DATA (EXPLICIT JOIN)
    # ===============================
    rows = (
        db.session.query(
            SalesTransaction.date.label("date"),
            Item.category.label("category"),
            SalesTransactionItem.quantity.label("quantity"),
        )
        .select_from(SalesTransaction)
        .join(
            SalesTransactionItem,
            SalesTransaction.id == SalesTransactionItem.transaction_id
        )
        .join(
            Item,
            Item.id == SalesTransactionItem.item_id
        )
        .all()
    )

    if not rows:
        print("[ML] No sales data found")
        return None

    df = pd.DataFrame(rows, columns=["date", "category", "quantity"])
    df["date"] = pd.to_datetime(df["date"]).dt.date

    # ===============================
    # 2️⃣ DAILY CATEGORY SERIES
    # ===============================
    daily = (
        df.groupby(["date", "category"])["quantity"]
        .sum()
        .unstack(fill_value=0)
        .sort_index()
    )

    print(f"[ML] Days of data: {len(daily)}")
    print(f"[ML] Categories: {daily.columns.tolist()}")

    if len(daily) < 5:
        print("[ML] Not enough data to train")
        return None

    # ===============================
    # 3️⃣ 🔑 LOG TRANSFORMATION (KEY FIX)
    # ===============================
    # log1p handles zeros safely
    log_daily = np.log1p(daily.values)

    # ===============================
    # 4️⃣ SCALE DATA
    # ===============================
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(log_daily)

    SEQ_LEN = min(30, len(scaled) - 1)

    # ===============================
    # 5️⃣ DATASET
    # ===============================
    class TSDataset(Dataset):
        def __init__(self, data, seq_len):
            self.X, self.y = [], []
            for i in range(len(data) - seq_len):
                self.X.append(data[i:i + seq_len])
                self.y.append(data[i + seq_len])

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return (
                torch.tensor(self.X[idx], dtype=torch.float32),
                torch.tensor(self.y[idx], dtype=torch.float32),
            )

    dataset = TSDataset(scaled, SEQ_LEN)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # ===============================
    # 6️⃣ MODEL
    # ===============================
    class LSTM(nn.Module):
        def __init__(self, features):
            super().__init__()
            self.lstm = nn.LSTM(features, 64, num_layers=2, batch_first=True)
            self.fc = nn.Linear(64, features)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1])

    model = LSTM(daily.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    # ===============================
    # 7️⃣ TRAIN
    # ===============================
    print("\n[ML] Training model...")
    for epoch in range(10):
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/10 | Loss: {total_loss:.4f}")

    # ===============================
    # 8️⃣ PREDICTION FUNCTION
    # ===============================
    def predict(days):
        model.eval()

        seq = torch.tensor(
            scaled[-SEQ_LEN:], dtype=torch.float32
        ).unsqueeze(0)

        preds = []

        for _ in range(days):
            with torch.no_grad():
                next_day = model(seq).numpy()[0]

            preds.append(next_day)

            seq = torch.cat(
                [seq[:, 1:, :], torch.tensor(next_day).view(1, 1, -1)],
                dim=1
            )

        # 🔄 INVERSE SCALE
        preds = scaler.inverse_transform(np.array(preds))

        # 🔄 INVERSE LOG TRANSFORM
        preds = np.expm1(preds)

        # Safety clamp
        return np.clip(preds, 0, None)

    # ===============================
    # 9️⃣ MAKE PREDICTIONS
    # ===============================
    pred_1 = predict(1)[0]
    pred_7 = predict(7)
    pred_30 = predict(30)

    print("\n===== PREDICTED TOMORROW =====")
    for i, cat in enumerate(daily.columns):
        print(f"{cat}: {int(round(pred_1[i]))} units")

    print("\n===== PREDICTED NEXT 7 DAYS (TOTAL) =====")
    for i, cat in enumerate(daily.columns):
        print(f"{cat}: {int(round(pred_7[:, i].sum()))} units")

    print("\n===== PREDICTED NEXT 30 DAYS (TOTAL) =====")
    for i, cat in enumerate(daily.columns):
        print(f"{cat}: {int(round(pred_30[:, i].sum()))} units")

    return {
        "tomorrow": dict(zip(daily.columns, pred_1.tolist())),
        "next_7_days": {
            cat: int(pred_7[:, i].sum())
            for i, cat in enumerate(daily.columns)
        },
        "next_30_days": {
            cat: int(pred_30[:, i].sum())
            for i, cat in enumerate(daily.columns)
        }
    }
