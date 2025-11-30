# ml/analyze_behavior.py
# Uses ONLY your CSV. No dummy generation. No output CSV saving.
# Time-series forecasts 7 days and 30 days, then prints TOP 5 best-selling categories for each horizon.
# User "next category" logic is frequency-based: suggests what each user buys most often.

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader


def run_ml():
    # ===============================
    # LOAD + DAILY STORE SERIES
    # ===============================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CSV_PATH = os.path.join(BASE_DIR, "tester2.csv")

    df = pd.read_csv(CSV_PATH)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

    # Build daily pivot (Date x Category = counts)
    daily = (
        df.groupby(["Date", "category"])
          .size()
          .unstack(fill_value=0)
          .sort_index()
    )
    # Ensure we "train all columns" present in the CSV (even if some are all zeros in the period)
    all_categories = sorted(df["category"].unique())
    daily = daily.reindex(columns=all_categories, fill_value=0)

    print("\nDaily Time-Series Shape:", daily.shape)
    print(daily.head())

    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(daily.values)

    # -------- Auto-adjust SEQ_LEN when data is short --------
    SEQ_LEN = 30
    if len(scaled_values) <= SEQ_LEN:
        SEQ_LEN = max(2, len(scaled_values) - 1)
        print(f"[WARN] Not enough days for SEQ_LEN=30. Using SEQ_LEN={SEQ_LEN} instead.")
    # --------------------------------------------------------

    class TimeSeriesDataset(Dataset):
        def __init__(self, data, seq_len=SEQ_LEN):
            self.X = []
            self.y = []
            for i in range(len(data) - seq_len):
                self.X.append(data[i:i+seq_len])
                self.y.append(data[i+seq_len])

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return (
                torch.tensor(self.X[idx], dtype=torch.float32),
                torch.tensor(self.y[idx], dtype=torch.float32),
            )

    dataset = TimeSeriesDataset(scaled_values, seq_len=SEQ_LEN)
    if len(dataset) == 0:
        raise ValueError(
            f"Time-series dataset is empty. Need more than {SEQ_LEN} days, "
            f"but got only {len(scaled_values)}."
        )

    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    num_categories_ts = daily.shape[1]

    class LSTMForecaster(nn.Module):
        def __init__(self, num_features, hidden_size=64, num_layers=2):
            super().__init__()
            self.lstm = nn.LSTM(
                num_features,
                hidden_size,
                num_layers,
                batch_first=True,
                dropout=0.2 if num_layers > 1 else 0.0,
            )
            self.fc = nn.Linear(hidden_size, num_features)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    model_ts = LSTMForecaster(num_categories_ts)
    criterion_ts = nn.MSELoss()
    optimizer_ts = torch.optim.Adam(model_ts.parameters(), lr=0.001)

    print("\nTraining Store-Level Time Series Model...\n")
    EPOCHS_TS = 10
    for epoch in range(EPOCHS_TS):
        total_loss = 0.0
        for xb, yb in loader:
            optimizer_ts.zero_grad()
            pred = model_ts(xb)
            loss = criterion_ts(pred, yb)
            loss.backward()
            optimizer_ts.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS_TS} Loss: {total_loss:.4f}")
    print("\nTime Series Training complete.\n")

    # ===============================
    # MULTI-STEP FORECAST FUNCTION
    # ===============================
    def forecast_days(model, scaled_series, days_ahead, seq_len):
        """
        Recursive multi-step forecasting.
        Returns (days_ahead, num_categories) in ORIGINAL (inverse-transformed) scale.
        """
        model.eval()
        preds_scaled = []

        current_seq = torch.tensor(
            scaled_series[-seq_len:], dtype=torch.float32
        ).unsqueeze(0)

        for _ in range(days_ahead):
            with torch.no_grad():
                next_scaled = model(current_seq).numpy()[0]
            preds_scaled.append(next_scaled)

            # slide window: drop first day, append predicted day
            next_scaled_t = torch.tensor(next_scaled, dtype=torch.float32).view(1, 1, -1)
            current_seq = torch.cat([current_seq[:, 1:, :], next_scaled_t], dim=1)

        preds_scaled = np.array(preds_scaled)
        preds = scaler.inverse_transform(preds_scaled)
        preds = np.clip(preds, 0, None)  # no negatives
        return preds

    # ===============================
    # SIMPLE OUTPUT: TOP 5 BEST-SELLING
    # ===============================
    def print_top5_items(preds, horizon_days, columns):
        totals = preds.sum(axis=0)  # sum over horizon per category
        top5_idx = np.argsort(totals)[::-1][:5]

        print(f"\n==================== TOP 5 BEST-SELLING (NEXT {horizon_days} DAYS) ====================\n")
        for rank, idx in enumerate(top5_idx, start=1):
            cat = columns[idx]
            total_sales = totals[idx]
            avg_per_day = total_sales / horizon_days
            print(f"{rank}. {cat} | Total: {total_sales:.2f} | Avg/day: {avg_per_day:.2f}")

    # 7-day forecast → print top 5 categories
    forecast_7 = forecast_days(model_ts, scaled_values, days_ahead=7, seq_len=SEQ_LEN)
    print_top5_items(forecast_7, 7, daily.columns)

    # 30-day forecast → print top 5 categories
    forecast_30 = forecast_days(model_ts, scaled_values, days_ahead=30, seq_len=SEQ_LEN)
    print_top5_items(forecast_30, 30, daily.columns)

    # ===============================
    # USER "OFTEN BUYS" RECOMMENDER (Frequency-based)
    # ===============================
    print("\n===============================================")
    print(" USER 'OFTEN BUYS' RECOMMENDER (FREQUENCY-BASED)")
    print("===============================================\n")

    df_sorted = df.sort_values(["Member_number", "Date"])

    def recommend_often_buys(df_sorted, top_k=3, print_users=30):
        results = []
        for user, group in df_sorted.groupby("Member_number", sort=False):
            counts = group["category"].value_counts()
            if counts.empty:
                continue

            max_count = counts.max()
            candidates = counts[counts == max_count].index.tolist()

            # tie-break: most recent among the tied top categories
            top_cat = candidates[0]
            if len(candidates) > 1:
                recent_cats = group.sort_values("Date")["category"].iloc[::-1]
                for c in recent_cats:
                    if c in candidates:
                        top_cat = c
                        break

            total = int(counts.sum())
            top_count = int(counts.loc[top_cat])
            confidence = top_count / total  # proportion of user purchases in that category
            top_list = counts.head(top_k).index.tolist()

            results.append({
                "user": user,
                "suggested_category": top_cat,
                "confidence": confidence,
                "support": top_count,
                "total": total,
                "top_k": top_list
            })

        # sort users: highest confidence then highest support
        results.sort(key=lambda x: (x["confidence"], x["support"]), reverse=True)

        print(f"==================== TOP {print_users} USERS (OFTEN-BUYS) ====================\n")
        for r in results[:print_users]:
            print(
                f"User {r['user']} | "
                f"Suggested (Most Often): {r['suggested_category']} | "
                f"Support: {r['support']}/{r['total']} ({r['confidence']*100:.2f}%) | "
                f"Top-{top_k}: {r['top_k']}"
            )
        return results

    # Print top 30 users by often-buys suggestion
    recommend_often_buys(df_sorted, top_k=3, print_users=30)


if __name__ == "__main__":
    run_ml()
