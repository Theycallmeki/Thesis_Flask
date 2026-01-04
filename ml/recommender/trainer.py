import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

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

    # ONLY block if there are no purchases at all
    if not interactions:
        state.model = None
        state.user_map = {}
        state.item_map = {}
        state.score_matrix = {}
        return

    # map ONLY users who interacted
    user_ids = list(interactions.keys())
    item_ids = [i.id for i in Item.query.all()]

    state.user_map = {uid: idx for idx, uid in enumerate(user_ids)}
    state.item_map = {iid: idx for idx, iid in enumerate(item_ids)}

    data = []
    for uid, items_ in interactions.items():
        uidx = state.user_map[uid]
        for iid, qty in items_.items():
            if iid in state.item_map:
                data.append((uidx, state.item_map[iid], float(qty)))

    loader = DataLoader(InteractionDataset(data), batch_size=32, shuffle=True)

    model = MFModel(len(state.user_map), len(state.item_map))
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for u, i, q in loader:
            opt.zero_grad()
            loss = loss_fn(model(u.long(), i.long()), q.float())
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
                score = (
                    model.user_emb.weight[uidx]
                    * model.item_emb.weight[iidx]
                ).sum().item()
                state.score_matrix[uid][iid] = score
