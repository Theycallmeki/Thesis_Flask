# ============================================================
# STORE B DATA NOTE:
# Store B is NOT yet operational and has NO available data.
# For this thesis, we pre-train the time-series model entirely
# on Store A's dataset: sales_with_categories_fast.csv.
#
# The model learns:
#   ✔ Seasonal grocery trends
#   ✔ Weekly buying patterns
#   ✔ Category-level demand over time
#
# Once Store B launches, this model will be fine-tuned using
# Store B's real sales using transfer learning.
# ============================================================

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

# ============================================================
# 1. LOAD STORE A DATASET
# ============================================================
df = pd.read_csv("sales_with_categories_fast.csv")

# Parse date (your CSV uses DD-MM-YYYY)
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

# Group by date + category → daily demand per category
daily = (
    df.groupby(["Date", "category"])
      .size()                         # number of items sold
      .unstack(fill_value=0)          # columns = categories
      .sort_index()                   # sorted by date
)

print("\nDaily Time-Series Shape:", daily.shape)
print(daily.head())

# ============================================================
# 2. NORMALIZE THE TIME SERIES
# ============================================================
scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(daily.values)

# ============================================================
# 3. CREATE SEQUENCES FOR LSTM TRAINING
# ============================================================
SEQ_LEN = 30  # use past 30 days to predict next day

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len=SEQ_LEN):
        self.data = data
        self.seq_len = seq_len
        self.X = []
        self.y = []

        for i in range(len(data) - seq_len):
            seq_x = data[i:i+seq_len]
            seq_y = data[i+seq_len]
            self.X.append(seq_x)
            self.y.append(seq_y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), \
               torch.tensor(self.y[idx], dtype=torch.float32)

dataset = TimeSeriesDataset(scaled_values)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

num_categories = daily.shape[1]

# ============================================================
# 4. DEFINE THE LSTM MODEL
# ============================================================
class LSTMForecaster(nn.Module):
    def __init__(self, num_features, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_features)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

model = LSTMForecaster(num_features=num_categories)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ============================================================
# 5. TRAIN THE MODEL (Store A only)
# ============================================================
EPOCHS = 20

print("\nTraining model on Store A data...\n")

for epoch in range(EPOCHS):
    total_loss = 0
    for xb, yb in loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}  Loss: {total_loss:.4f}")

print("\nTraining complete.\n")

# ============================================================
# 6. USE MODEL TO PREDICT NEXT DAY DEMAND
# ============================================================
with torch.no_grad():
    last_seq = torch.tensor(scaled_values[-SEQ_LEN:], dtype=torch.float32).unsqueeze(0)
    next_day_scaled = model(last_seq).numpy()[0]

next_day = scaler.inverse_transform(next_day_scaled.reshape(1, -1))[0]

future_df = pd.DataFrame({
    "category": daily.columns,
    "predicted_sales": next_day
}).sort_values("predicted_sales", ascending=False)

print("Predicted demand for tomorrow:")
print(future_df)
