"""
Trainer for the GIL-style separator (see sequence_generation/model/masked_separator.py).

Usage:
    python -m scripts.masked_separator_trainer \
        --config configs/enhancer_gosai_masked_separator.yaml \
        --out_dir ./runs/masked_separator

Alternating optimisation (GIL §3)
---------------------------------
Each epoch:
  (A) ENV INFERENCE STEP
        Pass the training set through the current separator to collect x_en
        embeddings, run K-means on them to get one env id per sample.
        This env assignment is frozen for the next update step.

  (B) SEPARATOR + Y-HEAD UPDATE STEP
        For each minibatch, look up the cached env ids and call
        model.compute_loss(x, clss, env_ids=...). The V-REx penalty
        variance across envs drives the separator to find a mask
        under which x_st -> y is invariant to the env (which is
        derived from x_en).

The very first epoch has no env labels yet (warmup): env_ids=None,
so L_inv is zero and the model behaves like ERM on x_st.
"""

import os
import copy
import argparse
import yaml

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
from ml_collections.config_dict import ConfigDict

from sequence_generation.utils.train_utils import load_dataloader, load_seed
from sequence_generation.model.masked_separator import MaskedSeparatorModel


# ---------------------------------------------------------------------------
# Simple K-means (PyTorch-only, no sklearn dependency)
# ---------------------------------------------------------------------------
@torch.no_grad()
def kmeans_fit_predict(
    X: torch.Tensor,
    K: int,
    n_iters: int = 20,
    seed: int = 0,
) -> torch.Tensor:
    """
    X: (N, D) feature matrix on any device.
    Returns: (N,) long tensor of cluster ids in [0, K).
    """
    N, D = X.shape
    device = X.device
    g = torch.Generator(device="cpu").manual_seed(seed)
    init_idx = torch.randperm(N, generator=g)[:K].to(device)
    centroids = X[init_idx].clone()                      # (K, D)

    for _ in range(n_iters):
        # assignment: (N, K) squared distances
        d2 = torch.cdist(X, centroids, p=2) ** 2
        assign = d2.argmin(dim=-1)                       # (N,)
        # update
        new_centroids = centroids.clone()
        for k in range(K):
            mask = assign == k
            if mask.sum() > 0:
                new_centroids[k] = X[mask].mean(dim=0)
        shift = (new_centroids - centroids).norm()
        centroids = new_centroids
        if shift < 1e-5:
            break
    return assign


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class MaskedSeparatorTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        load_seed(config.seed)
        self.train_loader, self.eval_loader, _ = load_dataloader(config)

        self.model = MaskedSeparatorModel(config).to(self.device)

        de_cfg = config.get("masked_separator", config.get("dual_encoder", {}))
        self.num_envs       = de_cfg.get("num_envs", 3)
        self.warmup_epochs  = de_cfg.get("warmup_epochs", 1)
        self.env_infer_every = de_cfg.get("env_infer_every", 1)   # epochs
        self.kmeans_iters   = de_cfg.get("kmeans_iters", 20)
        self.kmeans_samples = de_cfg.get("kmeans_samples", 8192)  # cap for speed

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.train.lr,
            weight_decay=config.optim.get("weight_decay", 0.0),
        )

        self.out_dir = config.get("out_dir", "./runs/masked_separator")
        os.makedirs(self.out_dir, exist_ok=True)

        # Frozen snapshot of the model used for env inference and per-batch
        # env assignment throughout an epoch. GIL assumes env labels are fixed
        # while the separator is being updated; we enforce this by deepcopying
        # the model at env-inference time and never updating the snapshot
        # until the next env-inference step.
        self._env_model = None
        self._env_centroids = None

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _infer_envs(self, epoch: int) -> torch.Tensor:
        """
        Sweep through the train loader once, collect x_en embeddings for up
        to `kmeans_samples` samples, fit K-means, and return a (N_total,)
        tensor of env ids (one per training sample, indexed in the order the
        loader yields them — we rely on shuffle=True elsewhere so we also
        need to key by original indices below).

        To stay simple and robust under DataLoader shuffling, we instead
        collect (batch_index_within_loader, env_id) pairs and rebuild a
        per-step env_id tensor on the fly during the update pass. This
        means: we do ONE pass to collect embeddings + cluster them, then
        re-embed during the update pass using *exactly the same* mini-batch
        order (we re-seed the shuffle).

        For simplicity we just collect embeddings in the order the loader
        produces them in this epoch, cluster them, and then the update pass
        runs *a separate* loader iteration. Since batches differ, we cannot
        reuse the clustering labels. Instead we embed x_en for the current
        batch at update time and assign each sample to the *nearest stored
        centroid*. The trainer therefore caches the centroids, not raw
        labels.
        """
        # Freeze a snapshot of the current model for the whole epoch, so
        # env assignments do not drift as the live separator is updated.
        self._env_model = copy.deepcopy(self.model).eval()
        for p in self._env_model.parameters():
            p.requires_grad_(False)

        feats = []
        collected = 0
        for batch in tqdm(self.train_loader, desc=f"epoch {epoch+1} [infer envs]"):
            x = batch["seqs"].to(self.device)
            B = x.size(0)
            _, _, x_en = self._env_model.separate(x, soft_input=False)
            h_en = self._env_model.embed_en(x_en)            # (B, H)
            feats.append(h_en)
            collected += B
            if collected >= self.kmeans_samples:
                break
        feats = torch.cat(feats, dim=0)[: self.kmeans_samples]
        assign = kmeans_fit_predict(feats, K=self.num_envs,
                                    n_iters=self.kmeans_iters, seed=epoch)
        # compute centroids from the assignment for downstream use
        centroids = torch.stack([
            feats[assign == k].mean(dim=0) if (assign == k).sum() > 0
            else feats.mean(dim=0)
            for k in range(self.num_envs)
        ], dim=0)                                            # (K, H)
        self._env_centroids = centroids.detach()
        # log env distribution
        counts = torch.bincount(assign, minlength=self.num_envs)
        print(f"[env infer] cluster sizes: {counts.tolist()}  "
              f"(of {feats.size(0)} sampled)")
        return centroids

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _assign_envs_for_batch(self, x: torch.Tensor) -> torch.Tensor:
        """Assign env id to each sample in the current batch using cached centroids."""
        # Use the frozen snapshot so env ids stay consistent with the
        # centroids that were fit at the start of this epoch.
        _, _, x_en = self._env_model.separate(x, soft_input=False)
        h_en = self._env_model.embed_en(x_en)                # (B, H)
        d2 = torch.cdist(h_en, self._env_centroids, p=2) ** 2
        return d2.argmin(dim=-1)                             # (B,)

    # ------------------------------------------------------------------
    def train(self):
        best = float("inf")
        num_epochs = self.config.train.num_epochs

        for epoch in range(num_epochs):
            # ---- (A) env inference step (skipped during warmup) --------
            envs_ready = False
            if epoch >= self.warmup_epochs and (epoch % self.env_infer_every == 0):
                self._infer_envs(epoch)
                envs_ready = True

            # ---- (B) separator + y-head update step --------------------
            self.model.train()
            agg = {"loss": 0.0, "L_sta": 0.0, "L_reg": 0.0,
                   "L_inv": 0.0, "L_smooth": 0.0,
                   "rho": 0.0, "mask_mean": 0.0}
            n = 0
            for batch in tqdm(self.train_loader, desc=f"epoch {epoch+1} [update]"):
                x    = batch["seqs"].to(self.device)
                clss = batch["clss"].to(self.device)

                # assign env ids using the cached centroids (if available)
                if self._env_model is not None and self._env_centroids is not None:
                    env_ids = self._assign_envs_for_batch(x)
                else:
                    env_ids = None

                self.optimizer.zero_grad()
                out = self.model.compute_loss(x, clss, env_ids=env_ids)
                out["loss"].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                for k in ("loss", "L_sta", "L_reg", "L_inv", "L_smooth",
                         "rho", "mask_mean"):
                    v = out[k]
                    agg[k] += float(v.item() if torch.is_tensor(v) else v)
                n += 1

            avg = {k: v / max(n, 1) for k, v in agg.items()}
            print(
                f"epoch {epoch+1:3d}  "
                f"loss={avg['loss']:.4f}  "
                f"L_sta={avg['L_sta']:.4f}  "
                f"L_inv={avg['L_inv']:.4f}  "
                f"L_reg={avg['L_reg']:.4f}  "
                f"L_smooth={avg['L_smooth']:.4f}  "
                f"rho={avg['rho']:.3f}  "
                f"mask_mean={avg['mask_mean']:.3f}"
            )

            if avg["loss"] < best:
                best = avg["loss"]
                ckpt = os.path.join(self.out_dir, "masked_separator_best.ckpt")
                torch.save({
                    "epoch": epoch,
                    "model": self.model.state_dict(),
                    "loss":  best,
                    "env_centroids": getattr(self, "_env_centroids", None),
                }, ckpt)
                print(f"  -> saved {ckpt}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = ConfigDict(yaml.safe_load(f))
    config.out_dir = args.out_dir

    MaskedSeparatorTrainer(config).train()


if __name__ == "__main__":
    main()