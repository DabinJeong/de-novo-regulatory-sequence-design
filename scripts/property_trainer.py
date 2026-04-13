"""
Joint property predictor + dual encoder trainer.

Trains a MaskedSeparatorModel (mask separator + y_head) so that:
  - y_head predicts activity from the invariant sub-sequence x_st
  - The separator learns which positions are functional (x_st) vs background (x_en)
  - V-REx invariance penalty ensures x_st predictions are robust across
    inferred environments (derived from x_en via K-means)

Loss:
    L = L_sta + lambda_reg * L_reg + beta_inv * L_inv

Usage:
    python -m scripts.main --config configs/enhancer_gosai.yaml \
                           --out_dir ./runs/property --train
"""

import os
import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from sequence_generation.utils.train_utils import load_dataloader, load_seed
from sequence_generation.model.masked_separator import MaskedSeparatorModel


# ---------------------------------------------------------------------------
# K-means (same as dual_encoder_trainer.py)
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
    centroids = X[init_idx].clone()

    for _ in range(n_iters):
        d2 = torch.cdist(X, centroids, p=2) ** 2
        assign = d2.argmin(dim=-1)
        new_centroids = centroids.clone()
        for k in range(K):
            mask = assign == k
            if mask.sum() > 0:
                new_centroids[k] = X[mask].mean(dim=0)
        shift = (new_centroids - centroids).norm()
        centroids = new_centroids
        if shift < 1e-5:
            break
    return assign, centroids


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        load_seed(config.seed)
        self.train_loader, self.eval_loader, _ = load_dataloader(config)

        self.model = MaskedSeparatorModel(config).to(self.device)

        de_cfg = config.get("masked_separator", config.get("dual_encoder", {}))
        self.num_envs = de_cfg.get("num_envs", 3)
        self.warmup_epochs = de_cfg.get("warmup_epochs", 1)
        self.env_infer_every = de_cfg.get("env_infer_every", 1)
        self.kmeans_iters = de_cfg.get("kmeans_iters", 20)
        self.kmeans_samples = de_cfg.get("kmeans_samples", 8192)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.train.lr,
            weight_decay=config.optim.get("weight_decay", 0.0),
        )

        self.out_dir = config.get("out_dir", "./runs/property")
        os.makedirs(self.out_dir, exist_ok=True)

        self._env_model = None
        self._env_centroids = None

    # ------------------------------------------------------------------
    # Environment inference (K-means on x_en embeddings)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _infer_envs(self, epoch: int) -> torch.Tensor:
        """Sweep training data, embed x_en, fit K-means, cache centroids."""
        self._env_model = copy.deepcopy(self.model).eval()
        for p in self._env_model.parameters():
            p.requires_grad_(False)

        feats = []
        alpha_max = float(self.config.model.alpha_max)
        collected = 0
        for batch in tqdm(self.train_loader, desc=f"epoch {epoch+1} [infer envs]"):
            x = batch["seqs"].to(self.device)
            B = x.size(0)
            t = torch.full((B,), alpha_max, device=self.device)
            _, _, x_en = self._env_model.separate(x, t=t, soft_input=False)
            h_en = self._env_model.embed_en(x_en, t)
            feats.append(h_en)
            collected += B
            if collected >= self.kmeans_samples:
                break
        feats = torch.cat(feats, dim=0)[: self.kmeans_samples]

        assign, centroids = kmeans_fit_predict(
            feats, K=self.num_envs, n_iters=self.kmeans_iters, seed=epoch,
        )
        self._env_centroids = centroids.detach()

        counts = torch.bincount(assign, minlength=self.num_envs)
        print(f"[env infer] cluster sizes: {counts.tolist()}  "
              f"(of {feats.size(0)} sampled)")
        return centroids

    @torch.no_grad()
    def _assign_envs_for_batch(self, x: torch.Tensor) -> torch.Tensor:
        """Assign env id to each sample using cached centroids."""
        B = x.size(0)
        alpha_max = float(self.config.model.alpha_max)
        t = torch.full((B,), alpha_max, device=self.device)
        _, _, x_en = self._env_model.separate(x, t=t, soft_input=False)
        h_en = self._env_model.embed_en(x_en, t)
        d2 = torch.cdist(h_en, self._env_centroids, p=2) ** 2
        return d2.argmin(dim=-1)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, loader):
        """Compute MSE, Pearson, Spearman on a data loader."""
        self.model.eval()
        alpha_max = float(self.config.model.alpha_max)
        all_y, all_y_hat = [], []

        for batch in loader:
            x = batch["seqs"].to(self.device)
            clss = batch["clss"].to(self.device)
            B = x.size(0)

            # Use clean one-hot (t = alpha_max) for eval
            t = torch.full((B,), alpha_max, device=self.device)
            M, x_st, _ = self.model.separate(x, t=t, soft_input=False)
            y_hat = self.model.predict_y_from_st(x_st, t)       # (B, 1)
            y = self.model._target(clss)                         # (B, 1)

            all_y.append(y.cpu())
            all_y_hat.append(y_hat.cpu())

        all_y = torch.cat(all_y, dim=0).squeeze(-1).numpy()
        all_y_hat = torch.cat(all_y_hat, dim=0).squeeze(-1).numpy()

        mse = float(np.mean((all_y - all_y_hat) ** 2))
        pcorr, _ = pearsonr(all_y, all_y_hat)
        scorr, _ = spearmanr(all_y, all_y_hat)

        return {"mse": mse, "pearson": pcorr, "spearman": scorr}

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def train(self):
        best_loss = float("inf")
        num_epochs = self.config.train.num_epochs

        for epoch in range(num_epochs):
            # ---- (A) env inference (skipped during warmup) ---------------
            if epoch >= self.warmup_epochs and (epoch % self.env_infer_every == 0):
                self._infer_envs(epoch)

            # ---- (B) separator + y_head update ---------------------------
            self.model.train()
            agg = {"loss": 0.0, "L_sta": 0.0, "L_reg": 0.0,
                   "L_inv": 0.0, "rho": 0.0, "mask_mean": 0.0}
            n = 0
            for batch in tqdm(self.train_loader, desc=f"epoch {epoch+1} [train]"):
                x = batch["seqs"].to(self.device)
                clss = batch["clss"].to(self.device)

                if self._env_model is not None and self._env_centroids is not None:
                    env_ids = self._assign_envs_for_batch(x)
                else:
                    env_ids = None

                self.optimizer.zero_grad()
                out = self.model.compute_loss(x, clss, env_ids=env_ids)
                out["loss"].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                for k in ("loss", "L_sta", "L_reg", "L_inv", "rho", "mask_mean"):
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
                f"rho={avg['rho']:.3f}  "
                f"mask_mean={avg['mask_mean']:.3f}"
            )

            # ---- (C) eval ------------------------------------------------
            eval_metrics = self.evaluate(self.eval_loader)
            print(
                f"  [eval] mse={eval_metrics['mse']:.4f}  "
                f"pearson={eval_metrics['pearson']:.4f}  "
                f"spearman={eval_metrics['spearman']:.4f}"
            )

            # ---- (D) checkpoint ------------------------------------------
            if avg["loss"] < best_loss:
                best_loss = avg["loss"]
                ckpt_path = os.path.join(self.out_dir, "best.ckpt")
                torch.save({
                    "epoch": epoch,
                    "model": self.model.state_dict(),
                    "loss": best_loss,
                    "eval_metrics": eval_metrics,
                    "env_centroids": self._env_centroids,
                }, ckpt_path)
                print(f"  -> saved {ckpt_path}")
