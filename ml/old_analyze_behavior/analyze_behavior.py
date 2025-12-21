# This is the analyze_behavior.py version used for testing LSTM and GRU model for Activity 2
# The file is intentionally commented

# import os
# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# from sklearn.preprocessing import MinMaxScaler
# from torch.utils.data import Dataset, DataLoader
# from sklearn.metrics import mean_squared_error

# # Load and prepare data
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# CSV_PATH = os.path.join(BASE_DIR, "sales_with_categories_fast.csv")

# df = pd.read_csv(CSV_PATH)
# df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

# daily = (
#     df.groupby(["Date", "category"])
#     .size()
#     .unstack(fill_value=0)
#     .sort_index()
# )

# print("\nDaily Time-Series Shape:", daily.shape)

# # Scale data
# scaler = MinMaxScaler()
# scaled_values = scaler.fit_transform(daily.values)

# SEQ_LEN = 30
# HORIZON = 7

# # Dataset
# class TimeSeriesDataset(Dataset):
#     def __init__(self, data, seq_len=SEQ_LEN, horizon=HORIZON):
#         self.X, self.y = [], []
#         for i in range(len(data) - seq_len - horizon + 1):
#             self.X.append(data[i : i + seq_len])
#             self.y.append(data[i + seq_len : i + seq_len + horizon])

#         self.X = np.array(self.X)
#         self.y = np.array(self.y)

#     def __len__(self):
#         return len(self.X)

#     def __getitem__(self, idx):
#         return (
#             torch.tensor(self.X[idx], dtype=torch.float32),
#             torch.tensor(self.y[idx], dtype=torch.float32),
#         )

# # Models
# class LSTMForecaster(nn.Module):
#     def __init__(self, num_features, hidden_size=64, num_layers=2):
#         super().__init__()
#         self.lstm = nn.LSTM(
#             num_features,
#             hidden_size,
#             num_layers,
#             batch_first=True
#         )
#         self.fc = nn.Linear(hidden_size, num_features * HORIZON)
#         self.num_features = num_features

#     def forward(self, x):
#         out, _ = self.lstm(x)
#         out = self.fc(out[:, -1, :])
#         return out.view(-1, HORIZON, self.num_features)

# class GRUForecaster(nn.Module):
#     def __init__(self, num_features, hidden_size=64, num_layers=2):
#         super().__init__()
#         self.gru = nn.GRU(
#             num_features,
#             hidden_size,
#             num_layers,
#             batch_first=True
#         )
#         self.fc = nn.Linear(hidden_size, num_features * HORIZON)
#         self.num_features = num_features

#     def forward(self, x):
#         out, _ = self.gru(x)
#         out = self.fc(out[:, -1, :])
#         return out.view(-1, HORIZON, self.num_features)

# # Training
# def train_model(model, dataset, epochs=50, lr=0.001):
#     loader = DataLoader(dataset, batch_size=32, shuffle=True)
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#     for epoch in range(epochs):
#         epoch_loss = 0
#         for xb, yb in loader:
#             optimizer.zero_grad()
#             preds = model(xb)
#             loss = criterion(preds, yb)
#             loss.backward()
#             optimizer.step()
#             epoch_loss += loss.item()

#         print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")

# # Prediction
# def predict(model, last_seq):
#     model.eval()
#     with torch.no_grad():
#         pred = model(last_seq)
#     return pred.numpy()[0]

# # RMSE calculations
# def rmse_7day(actual, predicted, categories):
#     """RMSE over full forecast horizon"""
#     return {
#         cat: np.sqrt(mean_squared_error(actual[:, i], predicted[:, i]))
#         for i, cat in enumerate(categories)
#     }

# def rmse_1day(actual, predicted, categories):
#     """RMSE using ONLY next-day prediction"""
#     return {
#         cat: np.sqrt(mean_squared_error(actual[0:1, i], predicted[0:1, i]))
#         for i, cat in enumerate(categories)
#     }

# # Run experiments
# num_features = daily.shape[1]
# last_seq = torch.tensor(
#     scaled_values[-SEQ_LEN:], dtype=torch.float32
# ).unsqueeze(0)

# models = {
#     "LSTM": LSTMForecaster(num_features),
#     "GRU": GRUForecaster(num_features),
# }

# dataset = TimeSeriesDataset(scaled_values)

# for name, model in models.items():
#     print(f"\n============================")
#     print(f" Training {name} Model")
#     print(f"============================\n")

#     train_model(model, dataset)

#     # Prediction 
#     pred_scaled = predict(model, last_seq)
#     predictions = scaler.inverse_transform(pred_scaled)

#     # Ground truth 
#     actual = daily.values[-HORIZON:]

#     # Metrics 
#     rmse_next_day = rmse_1day(actual, predictions, daily.columns)
#     rmse_7_days = rmse_7day(actual, predictions, daily.columns)

#     # Top 8 categories – NEXT DAY
#     top8_idx_1d = predictions[0].argsort()[::-1][:8]

#     print(f"\n{name} – Top 8 Categories (Next-Day Forecast)\n")

#     for idx in top8_idx_1d:
#         cat = daily.columns[idx]
#         print(
#             f"{cat}: "
#             f"Predicted Sales (Day 1) = {predictions[0, idx]:.2f}, "
#             f"RMSE (1-Day) = {rmse_next_day[cat]:.2f}"
#         )

#     # Top 8 categories – 7 DAYS
#     top8_idx_7d = predictions.sum(axis=0).argsort()[::-1][:8]

#     print(f"\n{name} – Top 8 Categories (7-Day Forecast)\n")

#     for idx in top8_idx_7d:
#         cat = daily.columns[idx]
#         print(
#             f"{cat}: "
#             f"Predicted Sales (7 Days) = {predictions[:, idx].sum():.2f}, "
#             f"RMSE (7-Day) = {rmse_7_days[cat]:.2f}"
#         )
