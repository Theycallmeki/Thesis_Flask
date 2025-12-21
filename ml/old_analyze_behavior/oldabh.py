# This is the oldest analyze_behavior.py version
# The file is intentionally commented


# import os
# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# from sklearn.preprocessing import MinMaxScaler, LabelEncoder
# from torch.utils.data import Dataset, DataLoader

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# CSV_PATH = os.path.join(BASE_DIR, "sales_with_categories_fast.csv")

# df = pd.read_csv(CSV_PATH)
# df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

# daily = (
#     df.groupby(["Date", "category"])
#       .size()
#       .unstack(fill_value=0)
#       .sort_index()
# )

# print("\nDaily Time-Series Shape:", daily.shape)
# print(daily.head())

# scaler = MinMaxScaler()
# scaled_values = scaler.fit_transform(daily.values)

# SEQ_LEN = 30

# class TimeSeriesDataset(Dataset):
#     def __init__(self, data, seq_len=SEQ_LEN):
#         self.X = []
#         self.y = []
#         for i in range(len(data) - seq_len):
#             self.X.append(data[i:i+seq_len])
#             self.y.append(data[i+seq_len])

#     def __len__(self):
#         return len(self.X)

#     def __getitem__(self, idx):
#         return torch.tensor(self.X[idx], dtype=torch.float32), \
#                torch.tensor(self.y[idx], dtype=torch.float32)

# dataset = TimeSeriesDataset(scaled_values)
# loader = DataLoader(dataset, batch_size=32, shuffle=True)

# num_categories = daily.shape[1]

# class LSTMForecaster(nn.Module):
#     def __init__(self, num_features, hidden_size=64, num_layers=2):
#         super().__init__()
#         self.lstm = nn.LSTM(num_features, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, num_features)

#     def forward(self, x):
#         out, _ = self.lstm(x)
#         return self.fc(out[:, -1, :])

# model_ts = LSTMForecaster(num_categories)
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model_ts.parameters(), lr=0.001)

# print("\nTraining Store-Level Time Series Model...\n")

# EPOCHS = 10
# for epoch in range(EPOCHS):
#     total_loss = 0
#     for xb, yb in loader:
#         optimizer.zero_grad()
#         pred = model_ts(xb)
#         loss = criterion(pred, yb)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     print(f"Epoch {epoch+1}/{EPOCHS} Loss: {total_loss:.4f}")

# print("\nTime Series Training complete.\n")

# with torch.no_grad():
#     last_seq = torch.tensor(scaled_values[-SEQ_LEN:], dtype=torch.float32).unsqueeze(0)
#     next_day_scaled = model_ts(last_seq).numpy()[0]

# next_day = scaler.inverse_transform(next_day_scaled.reshape(1, -1))[0]

# future_df = pd.DataFrame({
#     "category": daily.columns,
#     "predicted_sales": next_day
# }).sort_values("predicted_sales", ascending=False)

# print("\nPredicted demand for the next day:")
# print(future_df)

# print("\n===============================================")
# print(" TRAINING USER NEXT-PURCHASE MODEL")
# print("===============================================\n")

# df = df.sort_values(["Member_number", "Date"])

# le = LabelEncoder()
# df["cat_id"] = le.fit_transform(df["category"])

# user_sequences = {}
# for user, group in df.groupby("Member_number"):
#     user_sequences[user] = group["cat_id"].tolist()

# class UserSeqDataset(Dataset):
#     def __init__(self, sequences, seq_len=10):
#         self.samples = []
#         for seq in sequences.values():
#             if len(seq) < seq_len + 1:
#                 continue
#             for i in range(len(seq) - seq_len):
#                 self.samples.append((seq[i:i+seq_len], seq[i+seq_len]))

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         x, y = self.samples[idx]
#         return torch.tensor(x), torch.tensor(y)

# user_dataset = UserSeqDataset(user_sequences, seq_len=10)
# user_loader = DataLoader(user_dataset, batch_size=32, shuffle=True)

# num_categories = df["cat_id"].nunique()

# class NextCategoryModel(nn.Module):
#     def __init__(self, num_categories, embed_size=32, hidden_size=64):
#         super().__init__()
#         self.embed = nn.Embedding(num_categories, embed_size)
#         self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
#         self.fc = nn.Linear(hidden_size, num_categories)

#     def forward(self, x):
#         x = self.embed(x)
#         out, _ = self.lstm(x)
#         return self.fc(out[:, -1, :])

# model_user = NextCategoryModel(num_categories)

# criterion2 = nn.CrossEntropyLoss()
# optimizer2 = torch.optim.Adam(model_user.parameters(), lr=0.001)

# EPOCHS = 7
# for epoch in range(EPOCHS):
#     total_loss = 0
#     for xb, yb in user_loader:
#         optimizer2.zero_grad()
#         preds = model_user(xb)
#         loss = criterion2(preds, yb)
#         loss.backward()
#         optimizer2.step()
#         total_loss += loss.item()
#     print(f"User Model Epoch {epoch+1}/{EPOCHS} Loss: {total_loss:.4f}")

# print("\nUser-Level next-purchase model training complete!\n")

# all_user_predictions = []

# for user_id, history in user_sequences.items():
#     if len(history) < 10:
#         continue

#     last_seq = torch.tensor(history[-10:]).unsqueeze(0)

#     with torch.no_grad():
#         out = model_user(last_seq)
#         probs = torch.softmax(out, dim=1)[0]

#     top_prob, top_idx = torch.max(probs, dim=0)
#     predicted_category = le.inverse_transform([top_idx.item()])[0]

#     all_user_predictions.append({
#         "user": user_id,
#         "predicted_category": predicted_category,
#         "probability": float(top_prob.item())
#     })

# sorted_users = sorted(
#     all_user_predictions,
#     key=lambda x: x["probability"],
#     reverse=True
# )

# print("\n==================== TOP 10 USERS (NEXT CATEGORY PREDICTION) ====================\n")

# for entry in sorted_users[:1000]:
#     print(
#         f"User {entry['user']} | "
#         f"Next Category: {entry['predicted_category']} | "
#         f"Confidence: {entry['probability']:.4f}"
#     )
