import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from models.user import User
from models.item import Item

from .model import MFModel
from .dataset import build_interactions
from . import state

class InteractionDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def retrain_model(epochs=30):
    interactions = build_interactions()

    users = User.query.all()
    items = Item.query.all()
    if not interactions or not users or not items:
        return

    state.user_map = {u.id: i for i, u in enumerate(users)}
    state.item_map = {i.id: j for j, i in enumerate(items)}

    data = []
    for uid, items_ in interactions.items():
        for iid, qty in items_.items():
            data.append((
                state.user_map[uid],
                state.item_map[iid],
                float(qty)
            ))

    loader = DataLoader(InteractionDataset(data), batch_size=32, shuffle=True)

    model = MFModel(len(state.user_map), len(state.item_map))
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    for _ in range(epochs):
        for u, i, q in loader:
            u = u.long()          # user indices
            i = i.long()          # item indices
            q = q.float()         # target quantities
            opt.zero_grad()
            loss = loss_fn(model(u, i), q)
            loss.backward()
            opt.step()


    # build score matrix
    model.eval()
    state.model = model
    state.score_matrix = {}

    with torch.no_grad():
        for uid, uidx in state.user_map.items():
            state.score_matrix[uid] = {}
            for iid, iidx in state.item_map.items():
                score = (model.user_emb.weight[uidx] *
                         model.item_emb.weight[iidx]).sum().item()
                state.score_matrix[uid][iid] = score
