# ml/time_series_forecast.py
# STORE-LEVEL CATEGORY DEMAND FORECASTING (TIME-SERIES LSTM)
# PRINTS TOP-5 RESULTS LIKE ORIGINAL CODE

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader


def run_time_series_forecast():
    # ===============================
    # LOAD DATA
    # ===============================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CSV_PATH = os.path.join(BASE_DIR, "tester.csv")

    df = pd.read_csv(CSV_PATH)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

    # ===============================
    # BUILD DAILY CATEGORY SERIES
    # ===============================
    daily = (
        df.groupby(["Date", "category"])
          .size()
          .unstack(fill_value=0)
          .sort_index()
    )

    all_categories = sorted(df["category"].unique())
    daily = daily.reindex(columns=all_categories, fill_value=0)

    print("\nDaily series shape:", daily.shape)

    # ===============================
    # SCALE DATA
    # ===============================
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(daily.values)

    SEQ_LEN = 30
    if len(scaled) <= SEQ_LEN:
        SEQ_LEN = max(2, len(scaled) - 1)
        print(f"[WARN] Using SEQ_LEN={SEQ_LEN}")

    # ===============================
    # DATASET
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
    # LSTM MODEL
    # ===============================
    class LSTMForecaster(nn.Module):
        def __init__(self, features):
            super().__init__()
            self.lstm = nn.LSTM(
                features,
                64,
                num_layers=2,
                batch_first=True,
                dropout=0.2
            )
            self.fc = nn.Linear(64, features)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    model = LSTMForecaster(daily.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    # ===============================
    # TRAINING
    # ===============================
    print("\nTraining time-series model...\n")
    for epoch in range(10):
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[TS] Epoch {epoch+1}/10 Loss: {total_loss:.4f}")

    # ===============================
    # FORECAST FUNCTION
    # ===============================
    def forecast(days):
        model.eval()
        seq = torch.tensor(
            scaled[-SEQ_LEN:], dtype=torch.float32
        ).unsqueeze(0)

        preds = []
        for _ in range(days):
            with torch.no_grad():
                p = model(seq).numpy()[0]
            preds.append(p)
            seq = torch.cat(
                [seq[:, 1:, :], torch.tensor(p).view(1, 1, -1)],
                dim=1
            )

        preds = scaler.inverse_transform(np.array(preds))
        return np.clip(preds, 0, None)

    # ===============================
    # PRINT TOP-5 RESULTS (RESTORED)
    # ===============================
    def print_top5(preds, days, categories):
        totals = preds.sum(axis=0)
        top5_idx = np.argsort(totals)[::-1][:5]

        print(f"\n===== TOP 5 CATEGORIES (NEXT {days} DAYS) =====")
        for rank, idx in enumerate(top5_idx, 1):
            print(
                f"{rank}. {categories[idx]} | "
                f"Total: {totals[idx]:.2f} | "
                f"Avg/day: {totals[idx]/days:.2f}"
            )

    forecast_7 = forecast(7)
    forecast_30 = forecast(30)

    print_top5(forecast_7, 7, daily.columns)
    print_top5(forecast_30, 30, daily.columns)

    return {
        "7_days": forecast_7,
        "30_days": forecast_30,
        "categories": daily.columns.tolist()
    }
