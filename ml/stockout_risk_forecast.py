import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from db import db
from models.sales_transaction import SalesTransaction
from models.sales_transaction_item import SalesTransactionItem
from models.item import Item
from models.ai_stockout_risk import AIStockoutRisk


# -----------------------------
# LABEL ENCODING (ENUM SAFE)
# -----------------------------
LABEL_MAP = {"Low": 0, "Medium": 1, "High": 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


# -----------------------------
# DATASET
# -----------------------------
class StockoutDataset(Dataset):
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
class StockoutNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 16),
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
def run_stockout_risk_forecast():
    rows = (
        db.session.query(
            Item.id.label("item_id"),
            Item.name.label("item_name"),
            Item.category.label("category"),
            Item.quantity.label("current_stock"),
            SalesTransactionItem.quantity.label("sold_qty"),
        )
        .select_from(Item)
        .join(SalesTransactionItem, Item.id == SalesTransactionItem.item_id)
        .join(SalesTransaction, SalesTransaction.id == SalesTransactionItem.transaction_id)
        .all()
    )

    if not rows:
        return None

    df = pd.DataFrame(rows, columns=[
        "item_id",
        "item_name",
        "category",
        "current_stock",
        "sold_qty"
    ])

    features = []
    labels = []
    meta = []

    # -----------------------------
    # FEATURE ENGINEERING
    # -----------------------------
    for item_id, g in df.groupby("item_id"):
        total_sold = float(g["sold_qty"].sum())
        current_stock = int(g["current_stock"].iloc[0])

        initial_stock = total_sold + current_stock
        if initial_stock <= 0:
            continue

        sold_percentage = (total_sold / initial_stock) * 100
        remaining_percentage = 100 - sold_percentage

        # -----------------------------
        # FINAL PERCENT-BASED LABELING
        # 0–33  -> Low
        # 34–66 -> Medium
        # 67–100 -> High
        # -----------------------------
        if sold_percentage <= 33:
            label = "Low"
        elif sold_percentage <= 66:
            label = "Medium"
        else:
            label = "High"

        features.append([
            current_stock,
            sold_percentage,
            remaining_percentage
        ])
        labels.append(LABEL_MAP[label])

        meta.append({
            "item_id": int(item_id),
            "item_name": g["item_name"].iloc[0],
            "category": g["category"].iloc[0],
            "current_stock": current_stock,
            "avg_daily_sales": total_sold,          # reused field
            "days_of_stock_left": remaining_percentage
        })

    if len(features) < 5:
        return None

    dataset = StockoutDataset(features, labels)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    # -----------------------------
    # TRAIN MODEL
    # -----------------------------
    model = StockoutNet()
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
    AIStockoutRisk.query.delete()

    model.eval()
    with torch.no_grad():
        for x, info in zip(features, meta):
            logits = model(torch.tensor(x, dtype=torch.float32))
            pred = torch.argmax(logits).item()
            risk = INV_LABEL_MAP[pred]

            db.session.add(AIStockoutRisk(
                item_id=info["item_id"],
                item_name=info["item_name"],
                category=info["category"],
                current_stock=info["current_stock"],
                avg_daily_sales=info["avg_daily_sales"],
                days_of_stock_left=info["days_of_stock_left"],
                risk_level=risk
            ))

    db.session.commit()
    return True
