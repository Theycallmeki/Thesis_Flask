import sys
import os
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from db import db
from models.sales_transaction import SalesTransaction
from models.sales_transaction_item import SalesTransactionItem
from models.item import Item
from models.user import User

# Dataset
class UserItemDataset(Dataset):
    def __init__(self, interactions):
        self.data = interactions

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        u, i, q = self.data[idx]
        return torch.tensor(u, dtype=torch.long), torch.tensor(i, dtype=torch.long), torch.tensor(q, dtype=torch.float32)


# Model
class MFModel(nn.Module):
    def __init__(self, n_users, n_items, n_factors=16):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, n_factors)
        self.item_emb = nn.Embedding(n_items, n_factors)

    def forward(self, u, i):
        return (self.user_emb(u) * self.item_emb(i)).sum(dim=1)

# Build USER_ITEM_MATRIX
def build_user_item_matrix():
    user_item = defaultdict(lambda: defaultdict(int))
    transactions = SalesTransaction.query.all()
    for tx in transactions:
        for ti in tx.items:
            user_item[tx.user_id][ti.item_id] += ti.quantity
    return user_item


# Train CF model

def train_cf_model(epochs=100, n_factors=16):
    user_item_matrix = build_user_item_matrix()
    if not user_item_matrix:
        print("No sales transactions found.")
        return {}

    all_users = User.query.all()
    all_items = Item.query.all()
    if not all_users or not all_items:
        print("No users or items found in DB.")
        return {}

    user_map = {user.id: idx for idx, user in enumerate(all_users)}
    item_map = {item.id: idx for idx, item in enumerate(all_items)}

    # prepare interactions
    interactions = []
    for uid, items in user_item_matrix.items():
        for iid, qty in items.items():
            interactions.append((user_map[uid], item_map[iid], qty))

    print(f"Users in DB: {len(all_users)}")
    print(f"Items in DB: {len(all_items)}")
    print(f"Interactions to train on: {len(interactions)}")

    if not interactions:
        print("No interactions to train on.")
        return {}

    dataset = UserItemDataset(interactions)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = MFModel(len(user_map), len(item_map), n_factors)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for u, i, q in loader:
            optimizer.zero_grad()
            pred = model(u, i)
            loss = loss_fn(pred, q)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs} | Loss: {total_loss:.4f}")

    # build USER_ITEM_MATRIX from trained embeddings
    model.eval()
    trained_matrix = defaultdict(lambda: defaultdict(float))
    with torch.no_grad():
        for uid in user_map:
            u_idx = user_map[uid]
            u_vec = model.user_emb(torch.tensor(u_idx))
            for iid in item_map:
                i_idx = item_map[iid]
                i_vec = model.item_emb(torch.tensor(i_idx))
                trained_matrix[uid][iid] = float((u_vec * i_vec).sum())

    print("Training complete!")
    return trained_matrix


# Recommend products
def recommend_products(user_id, top_n=5, user_item_matrix=None):
    if not User.query.get(user_id):
        return []
    if user_item_matrix is None:
        user_item_matrix = train_cf_model(epochs=100)
    if user_id not in user_item_matrix or not user_item_matrix[user_id]:
        return []
    ranked_items = sorted(user_item_matrix[user_id].items(), key=lambda x: x[1], reverse=True)
    item_ids = [iid for iid, _ in ranked_items[:top_n]]
    if not item_ids:
        return []
    return Item.query.filter(Item.id.in_(item_ids)).all()

# CLI Test
if __name__ == "__main__":
    import time
    from app import app

    with app.app_context():
        print("Training CF model based on current user sales data...")
        start_time = time.time()

        matrix = train_cf_model(epochs=100)

        elapsed = time.time() - start_time
        print(f"Training took {elapsed:.2f} seconds")

        for user in User.query.all():
            items = recommend_products(user.id, top_n=5, user_item_matrix=matrix)
            if not items:
                print(f"No recommendations for user {user.id}")
            else:
                print(f"Recommendations for user {user.id}:")
                for item in items:
                    print(f"- {item.name} ({item.category}) - ${item.price}")
