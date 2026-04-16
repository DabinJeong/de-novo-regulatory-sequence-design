"""
Trainer for the PropertyScorer (proposal: slide 14-15).

Trains K independent SeqRegressor members with different seeds and saves a
single checkpoint that GuidedSampler can load.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from sequence_generation.utils.train_utils import load_dataloader, load_seed
from sequence_generation.model.property_scorer import PropertyScorer


class EnsembleTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ens_cfg = config.get("ensemble", {})
        self.num_members = ens_cfg.get("num_members", 5)
        self.target_idx  = ens_cfg.get("target_idx", 0)   # which task in clss

        self.train_loader, self.eval_loader, _ = load_dataloader(config)

        self.model = PropertyScorer(
            alphabet_size=config.model.alphabet_size,
            num_members=self.num_members,
            hidden_dim=ens_cfg.get("hidden_dim", 128),
            depth=ens_cfg.get("depth", 4),
            dropout=ens_cfg.get("dropout", 0.1),
        ).to(self.device)

        # one optimiser per member, with a different seed before init below
        self.optimizers = [
            torch.optim.AdamW(m.parameters(), lr=config.train.lr,
                              weight_decay=config.optim.get("weight_decay", 0.0))
            for m in self.model.members
        ]

        self.out_dir = config.get("out_dir", "./runs/ensemble")
        os.makedirs(self.out_dir, exist_ok=True)

    def _step(self, batch, member_idx):
        m = self.model.members[member_idx]
        seqs = batch["seqs"].to(self.device)
        y    = batch["clss"][:, self.target_idx:self.target_idx + 1].float().to(self.device)
        pred = m({"seqs": seqs})
        return F.mse_loss(pred, y)

    def train(self):
        num_epochs = self.config.train.num_epochs
        for member_idx in range(self.num_members):
            # different seed per member -> different init / data order
            load_seed(self.config.seed + 1000 * member_idx)
            print(f"\n=== Training ensemble member {member_idx + 1}/{self.num_members} ===")
            for epoch in range(num_epochs):
                self.model.members[member_idx].train()
                losses = []
                for batch in tqdm(self.train_loader, desc=f"member {member_idx} epoch {epoch+1}"):
                    self.optimizers[member_idx].zero_grad()
                    loss = self._step(batch, member_idx)
                    loss.backward()
                    self.optimizers[member_idx].step()
                    losses.append(loss.item())
                print(f"  member {member_idx} epoch {epoch+1}  train_mse={sum(losses)/len(losses):.4f}")

        ckpt_path = os.path.join(self.out_dir, "ensemble_best.ckpt")
        torch.save({"model": self.model.state_dict()}, ckpt_path)
        print(f"\nSaved ensemble checkpoint -> {ckpt_path}")
