import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datetime import date

from db import db
from models.sales_transaction import SalesTransaction
from models.sales_transaction_item import SalesTransactionItem
from models.item import Item
from models.ai_item_movement import AIItemMovement


# -----------------------------
# LABEL ENCODING
# -----------------------------
LABEL_MAP = {"Slow": 0, "Medium": 1, "Fast": 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


# -----------------------------
# DATASET
# -----------------------------
class MovementDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -----------------------------
# MODEL
# -----------------------------
class MovementNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------
# TRAIN + PREDICT
# -----------------------------
def run_item_movement_forecast():
    rows = (
        db.session.query(
            Item.id.label("item_id"),
            Item.name.label("item_name"),
            Item.category.label("category"),
            SalesTransaction.date.label("date"),
            SalesTransactionItem.quantity.label("quantity"),
        )
        .select_from(Item)
        .join(SalesTransactionItem, Item.id == SalesTransactionItem.item_id)
        .join(SalesTransaction, SalesTransaction.id == SalesTransactionItem.transaction_id)
        .all()
    )

    if not rows:
        return None

    df = pd.DataFrame(rows, columns=[
        "item_id", "item_name", "category", "date", "quantity"
    ])
    df["date"] = pd.to_datetime(df["date"]).dt.date

    today = date.today()

    features = []
    labels = []
    meta = []

    # -----------------------------
    # FEATURE ENGINEERING
    # -----------------------------
    for item_id, g in df.groupby("item_id"):
        days_active = (g["date"].max() - g["date"].min()).days + 1
        avg_daily_sales = float(g["quantity"].sum() / max(days_active, 1))
        days_since_last_sale = int((today - g["date"].max()).days)

        # 👇 temporary heuristic labels for training
        if avg_daily_sales >= 5 and days_since_last_sale <= 3:
            label = "Fast"
        elif avg_daily_sales >= 2 and days_since_last_sale <= 7:
            label = "Medium"
        else:
            label = "Slow"

        features.append([avg_daily_sales, days_since_last_sale])
        labels.append(LABEL_MAP[label])

        meta.append({
            "item_id": int(item_id),
            "item_name": g["item_name"].iloc[0],
            "category": g["category"].iloc[0],
            "avg_daily_sales": avg_daily_sales,
            "days_since_last_sale": days_since_last_sale
        })

    if len(features) < 5:
        return None

    dataset = MovementDataset(features, labels)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    # -----------------------------
    # TRAIN MODEL
    # -----------------------------
    model = MovementNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(50):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

    # -----------------------------
    # SAVE PREDICTIONS
    # -----------------------------
    AIItemMovement.query.delete()

    model.eval()
    with torch.no_grad():
        for x, info in zip(features, meta):
            logits = model(torch.tensor(x, dtype=torch.float32))
            pred = torch.argmax(logits).item()
            label = INV_LABEL_MAP[pred]

            db.session.add(AIItemMovement(
                item_id=info["item_id"],
                item_name=info["item_name"],
                category=info["category"],
                avg_daily_sales=float(info["avg_daily_sales"]),
                days_since_last_sale=int(info["days_since_last_sale"]),
                movement_class=label
            ))

    db.session.commit()
    return True
