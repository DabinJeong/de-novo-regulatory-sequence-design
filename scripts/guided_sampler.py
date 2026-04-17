"""
Inference-time guided Dirichlet FM sampler — **GIL edition**.

Velocity field:

    u*(x_t) = u_t(x_t)                              # frozen Dirichlet FM backbone
            + alpha * grad mu_hat(x_t)              # property guidance (ensemble)
            - beta  * grad sigma2_hat(x_t)          # uncertainty penalty
            + eta   * grad y_st(x_t)                # x_st should predict high y
            + xi    * sqrt(lambda) * grad_en_push(x_t)  # x_en pushed off known envs

The separator is trained under GIL-style V-REx invariance, so:
  * y_st is guaranteed (modulo training error) to be env-invariant, hence
    pulling x_st towards high y does not interact with background shift;
  * x_en contains environment-specific information by construction, hence
    pushing it away from the *nearest* training env centroid is a principled
    realisation of MOOD-style OOD novelty control.

At inference we:
    M     = separator(xt, t)
    x_st  = xt * M
    x_en  = xt * (1 - M)
    grad_y_st    = d/dxt  y_head(x_st, t)
    grad_en_push = d/dxt  min_k || embed_en(x_en, t) - centroid_k ||^2

`env_centroids` are loaded from the masked-separator checkpoint (the trainer
saves them alongside the model weights after the final K-means run).

The backbone weights are NOT updated. All steering happens at solve time.

Modes:
    lambda = 0  -> in-distribution sampling
    lambda -> 1 -> stronger environmental-background push (OOD via v-shift)
"""

import os
import csv
import torch
import torch.nn.functional as F

from sequence_generation.utils.flow_utils import (
    DirichletConditionalFlow, expand_simplex,
)
from sequence_generation.utils.train_utils import load_generator, load_dataloader
from sequence_generation.model.property_scorer import PropertyScorer
from sequence_generation.model.masked_separator import MaskedSeparatorModel


INT_TO_BASE = {0: "A", 1: "C", 2: "G", 3: "T"}


class GuidedSampler:
    """Frozen Dirichlet FM backbone + PropertyScorer + GIL separator guidance."""

    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---- frozen Dirichlet FM backbone ---------------------------------
        self.model, _, _ = load_generator(config)
        self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.K = config.model.alphabet_size
        self.condflow = DirichletConditionalFlow(
            K=self.K,
            alpha_spacing=0.01,
            alpha_max=config.model.alpha_max,
        )

        # ---- PropertyScorer (mu_hat, sigma2_hat) ----------------------------
        ens_cfg = config.get("ensemble", {})
        self.scorer = PropertyScorer(
            alphabet_size=self.K,
            num_members=ens_cfg.get("num_members", 5),
            hidden_dim=ens_cfg.get("hidden_dim", 128),
            depth=ens_cfg.get("depth", 4),
            dropout=ens_cfg.get("dropout", 0.1),
        )
        ens_ckpt = ens_cfg.get("checkpoint_path", None)
        if ens_ckpt and os.path.exists(ens_ckpt):
            sd = torch.load(ens_ckpt, map_location="cpu")
            self.scorer.load_state_dict(sd.get("model", sd))
            print(f"[GuidedSampler] loaded PropertyScorer checkpoint from {ens_ckpt}")
        else:
            raise FileNotFoundError(
                f"PropertyScorer checkpoint required but not found at {ens_ckpt}. "
                "Train with `scripts.main_guided --train_ensemble` first."
            )
        self.scorer.to(self.device).eval()
        for p in self.scorer.parameters():
            p.requires_grad_(False)

        # ---- GIL-style separator ------------------------------------------
        self.separator_model = MaskedSeparatorModel(config)
        de_cfg = config.get("masked_separator", config.get("dual_encoder", {}))
        de_ckpt = de_cfg.get("checkpoint_path", None)
        if de_ckpt and os.path.exists(de_ckpt):
            ckpt = torch.load(de_ckpt, map_location="cpu")
            self.separator_model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
            print(f"[GuidedSampler] loaded masked-separator checkpoint from {de_ckpt}")
            # Try to load env centroids saved by the trainer
            env_centroids = ckpt.get("env_centroids", None) if isinstance(ckpt, dict) else None
        else:
            raise FileNotFoundError(
                f"Masked-separator checkpoint required but not found at {de_ckpt}. "
                "Train with `scripts.masked_separator_trainer` first."
            )
        self.separator_model.to(self.device).eval()
        for p in self.separator_model.parameters():
            p.requires_grad_(False)

        # ---- guidance hyper-parameters ------------------------------------
        s = config.sampling
        self.alpha_g  = s.get("alpha_guidance", 1.0)     # property weight
        self.beta_g   = s.get("beta_uncertainty", 1.0)   # uncertainty penalty
        self.eta_st   = s.get("eta_stable", 1.0)         # stable sufficiency push
        self.xi_en    = s.get("xi_en_push", 1.0)         # env-push (v-shift)
        self.lam      = s.get("ood_lambda", 0.0)         # 0..1 novelty knob
        self.gamma_r  = s.get("gamma_rank", 1.0)         # ranking penalty on sigma
        self.delta_r  = s.get("delta_rank", 0.0)         # ranking bonus on novelty
        self.n_steps  = s.get("n_steps", 128)
        self.flow_temp = s.get("flow_temp", 1.0)
        self.prior_pseudocount = s.get("prior_pseudocount", 0.1)

        # ---- bounded guidance (position-wise grad mask + soft leash) ------
        # Drawn fresh per batch. When disabled, the sampler behaves exactly
        # as before. Reference sequences anchor x_st positions so that
        # activity-optimizing edits concentrate on x_en (env) positions.
        bg_cfg = s.get("bounded_guidance", {})
        self.bg_enabled          = bg_cfg.get("enabled", False)
        self.bg_reference_source = bg_cfg.get("reference_source", "dataset")
        self.bg_reference_path   = bg_cfg.get("reference_path", None)
        self.bg_reference_split  = bg_cfg.get("reference_split", "train")
        self.bg_grad_mask_mode   = bg_cfg.get("grad_mask_mode", "none")
        self.bg_leash_weight     = float(bg_cfg.get("leash_weight", 0.0))
        self._bg_ref_pool = None     # (N, L) long tensor, lazily loaded

        # ---- env centroids: either from checkpoint or recompute ----------
        if env_centroids is not None:
            self.env_centroids = env_centroids.to(self.device)
            print(f"[GuidedSampler] loaded {self.env_centroids.size(0)} env centroids "
                  f"from masked-separator checkpoint (dim={self.env_centroids.size(-1)})")
        else:
            print("[GuidedSampler] no env_centroids in checkpoint; recomputing "
                  "from training data (single-centroid fallback).")
            self.env_centroids = self._precompute_en_centroid(
                n_samples=de_cfg.get("n_en_centroid_samples", 2048),
            )

    # ----------------------------------------------------------------------
    # fallback: single global centroid if checkpoint didn't save per-env ones
    # ----------------------------------------------------------------------
    @torch.no_grad()
    def _precompute_en_centroid(self, n_samples: int) -> torch.Tensor:
        train_loader, _, _ = load_dataloader(self.config)
        collected = 0
        feats = []
        for batch in train_loader:
            x = batch["seqs"].to(self.device)
            B = x.size(0)
            _, _, x_en = self.separator_model.separate(x, soft_input=False)
            h_en = self.separator_model.embed_en(x_en)           # (B, H)
            feats.append(h_en)
            collected += B
            if collected >= n_samples:
                break
        feats = torch.cat(feats, dim=0)
        centroid = feats.mean(dim=0, keepdim=True)           # (1, H)
        return centroid.detach()

    # ----------------------------------------------------------------------
    # Reference pool for bounded guidance / asymmetric init
    # ----------------------------------------------------------------------
    @torch.no_grad()
    def _load_reference_pool(self, cap: int = 8192) -> torch.Tensor:
        """Return a cached (N, L) long tensor of reference token ids."""
        if self._bg_ref_pool is not None:
            return self._bg_ref_pool

        if self.bg_reference_source == "file" and self.bg_reference_path:
            import pandas as pd
            df = pd.read_csv(self.bg_reference_path)
            col = "seq" if "seq" in df.columns else df.columns[0]
            base_to_int = {"A": 0, "C": 1, "G": 2, "T": 3}
            ids = torch.tensor(
                [[base_to_int[c] for c in s] for s in df[col].tolist()],
                dtype=torch.long,
            )
        else:
            train_loader, val_loader, test_loader = load_dataloader(self.config)
            loader = {
                "train": train_loader, "val": val_loader, "test": test_loader,
            }[self.bg_reference_split]
            chunks = []
            collected = 0
            for batch in loader:
                chunks.append(batch["seqs"])
                collected += batch["seqs"].size(0)
                if collected >= cap:
                    break
            ids = torch.cat(chunks, dim=0)[:cap]

        self._bg_ref_pool = ids.to(self.device)
        print(f"[GuidedSampler] bounded-guidance reference pool: "
              f"{self._bg_ref_pool.size(0)} seqs "
              f"(source={self.bg_reference_source}, split={self.bg_reference_split})")
        return self._bg_ref_pool

    @torch.no_grad()
    def _draw_reference_batch(self, B: int):
        """Sample B reference seqs; return (x_ref_onehot, M_ref)."""
        pool = self._load_reference_pool()
        idx = torch.randint(0, pool.size(0), (B,), device=self.device)
        x_ref_ids = pool[idx]                                            # (B, L)
        x_ref = F.one_hot(x_ref_ids.long(), num_classes=self.K).float()  # (B, L, K)
        M_ref, _, _ = self.separator_model.separate(x_ref_ids, soft_input=False)
        return x_ref, M_ref                                              # (B,L,K),(B,L)

    # ----------------------------------------------------------------------
    # PropertyScorer gradients (property + uncertainty)
    # ----------------------------------------------------------------------
    def _mu_sigma2_grads(self, xt: torch.Tensor):
        """
        Returns (grad_mu, grad_sigma2, mu, sigma2), each grad shape (B,L,K).

        NOTE: the manual forward here bypasses SeqRegressor's embedder by
        feeding a soft-mixture embedding. It must stay in sync with
        SeqRegressor's architecture. Outstanding review item: #1.
        """
        B, L, K = xt.shape
        x = xt.detach().clone().requires_grad_(True)

        preds = []
        for m in self.scorer.members:
            W = m.embedder.weight
            soft_emb = torch.einsum("blk,kh->blh", x, W)
            h = soft_emb.transpose(1, 2)
            xs = [soft_emb]
            for conv in m.convs:
                h = torch.tanh(conv(h))
                xs.append(h.transpose(1, 2))
            feat = torch.cat(xs, dim=-1)
            gate = m.sigmoid_linear(feat) * m.tanh_linear(feat)
            pooled = torch.tanh(gate.mean(dim=1))
            preds.append(m.final_linear(pooled))
        preds = torch.stack(preds, dim=0)                    # (K_ens, B, 1)
        mu = preds.mean(0)
        var = preds.var(0, unbiased=False)

        grad_mu = torch.autograd.grad(mu.sum(), x, retain_graph=True)[0]
        grad_var = torch.autograd.grad(var.sum(), x, retain_graph=False)[0]

        grad_mu  = grad_mu  - grad_mu.mean(-1, keepdim=True)
        grad_var = grad_var - grad_var.mean(-1, keepdim=True)
        return grad_mu.detach(), grad_var.detach(), mu.detach(), var.detach()

    # ----------------------------------------------------------------------
    # GIL separator gradients (stable-sufficiency + env-push)
    # ----------------------------------------------------------------------
    def _gil_grads(self, xt: torch.Tensor):
        """
        Returns (grad_stable, grad_en_push), each (B, L, K).

        grad_stable  : direction increasing y_head(x_st)
        grad_en_push : direction increasing the min-distance from h_en(x_en)
                       to the set of training env centroids
        """
        x = xt.detach().clone().requires_grad_(True)

        M = self.separator_model.separator(x)                       # (B, L)
        M3 = M.unsqueeze(-1)
        bkg = self.separator_model.mask_embed.view(1, 1, -1)        # (1, 1, K)
        x_st = x * M3 + bkg * (1.0 - M3)
        x_en = x * (1.0 - M3) + bkg * M3

        # (1) stable-sufficiency
        y_st = self.separator_model.predict_y_from_st(x_st)         # (B, 1)
        grad_stable = torch.autograd.grad(
            y_st.sum(), x, retain_graph=True,
        )[0]

        # (2) env push: distance to NEAREST env centroid
        h_en = self.separator_model.embed_en(x_en)                  # (B, H)
        #   (B, H) vs (K, H)  ->  (B, K) squared distances
        d2 = torch.cdist(h_en, self.env_centroids, p=2) ** 2       # (B, K)
        nearest = d2.min(dim=-1).values                            # (B,)
        en_obj = nearest.mean()
        grad_en_push = torch.autograd.grad(
            en_obj, x, retain_graph=False,
        )[0]

        grad_stable  = grad_stable  - grad_stable.mean(-1, keepdim=True)
        grad_en_push = grad_en_push - grad_en_push.mean(-1, keepdim=True)
        return grad_stable.detach(), grad_en_push.detach()

    # ----------------------------------------------------------------------
    # main sampling loop
    # ----------------------------------------------------------------------
    def sample(self, B: int, L: int):
        device = self.device
        K = self.K

        xt = torch.distributions.Dirichlet(
            torch.ones(B, L, K, device=device)
        ).sample()
        eye = torch.eye(K, device=device)

        # ---- bounded guidance: draw reference batch once per sample() ----
        if self.bg_enabled:
            x_ref, M_ref = self._draw_reference_batch(B)           # (B,L,K),(B,L)
            M_ref3 = M_ref.unsqueeze(-1)                           # (B,L,1)
        else:
            x_ref = None
            M_ref3 = None

        t_span = torch.linspace(
            1.0, self.config.model.alpha_max,
            steps=self.n_steps, device=device,
        )

        for s, t in zip(t_span[:-1], t_span[1:]):
            xt_exp, _ = expand_simplex(xt, s[None].expand(B), self.prior_pseudocount)

            with torch.no_grad():
                logits = self.model(xt_exp, t=t[None].expand(B))
                flow_probs = F.softmax(logits / self.flow_temp, dim=-1)

            # ---- backbone velocity field u_t(xt) --------------------------
            c_factor = self.condflow.c_factor(xt.cpu().numpy(), s.item())
            c_factor = torch.from_numpy(c_factor).to(xt)
            cond_flows = (eye - xt.unsqueeze(-1)) * c_factor.unsqueeze(-2)
            u_t = (flow_probs.unsqueeze(-2) * cond_flows).sum(-1)

            # ---- PropertyScorer guidance -----------------------------------
            grad_mu, grad_var, _, _ = self._mu_sigma2_grads(xt)

            # ---- GIL separator guidance -----------------------------------
            grad_stable, grad_en_push = self._gil_grads(xt)

            # ---- bounded guidance: position-wise gradient masking --------
            # Keeps property/uncertainty gradients from acting on one region.
            # "en_only" implements the "conservative x_st, free x_en" design.
            if self.bg_enabled and self.bg_grad_mask_mode != "none":
                if self.bg_grad_mask_mode == "en_only":
                    gmask = 1.0 - M_ref3
                elif self.bg_grad_mask_mode == "st_only":
                    gmask = M_ref3
                else:
                    raise ValueError(
                        f"bounded_guidance.grad_mask_mode must be "
                        f"'none'|'st_only'|'en_only', got {self.bg_grad_mask_mode!r}"
                    )
                grad_mu  = grad_mu  * gmask
                grad_var = grad_var * gmask

            u_star = (
                u_t
                + self.alpha_g * grad_mu
                - self.beta_g  * grad_var
                + self.eta_st  * grad_stable
                + self.xi_en * (self.lam ** 0.5) * grad_en_push
            )

            # ---- bounded guidance: soft quadratic leash on x_st ----------
            # L_leash = || M_ref * (xt - x_ref) ||^2 (position-wise weight).
            # Gradient pulls xt toward x_ref only where M_ref is high.
            if self.bg_enabled and self.bg_leash_weight > 0.0:
                leash_grad = 2.0 * M_ref3 * (xt - x_ref)
                leash_grad = leash_grad - leash_grad.mean(-1, keepdim=True)
                u_star = u_star - self.bg_leash_weight * leash_grad

            xt = (xt + u_star * (t - s)).clamp(min=1e-8)
            xt = xt / xt.sum(-1, keepdim=True)

        seq_pred = xt.argmax(-1)
        return seq_pred, xt

    # ----------------------------------------------------------------------
    # ranking: score(x) = mu_hat - gamma * sigma_hat + delta * novelty(x)
    # ----------------------------------------------------------------------
    @torch.no_grad()
    def score_and_rank(self, seqs: torch.Tensor, ref_seqs: torch.Tensor = None):
        mu, var = self.scorer.mu_sigma2({"seqs": seqs.to(self.device)})
        mu = mu.squeeze(-1)
        sigma = var.clamp_min(1e-12).sqrt().squeeze(-1)

        if ref_seqs is not None and ref_seqs.numel() > 0:
            r = ref_seqs.to(self.device)
            d = (seqs.to(self.device).unsqueeze(1) != r.unsqueeze(0)).float().mean(-1)
            novelty = d.min(dim=1).values
        else:
            novelty = torch.zeros_like(mu)

        score = mu - self.gamma_r * sigma + self.delta_r * novelty
        return {"score": score, "mu": mu, "sigma": sigma, "novelty": novelty}


# ===========================================================================
# CLI driver
# ===========================================================================
def main():
    import argparse, yaml
    from ml_collections.config_dict import ConfigDict

    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--num_batches", type=int, default=2)
    args = parser.parse_args()

    with open(args.config) as f:
        config = ConfigDict(yaml.safe_load(f))
    config.out_dir = args.out_dir
    os.makedirs(args.out_dir, exist_ok=True)

    sampler = GuidedSampler(config)
    B = config.loader.eval_batch_size
    L = config.dataset.seq_length

    out_path = os.path.join(args.out_dir, "guided_sequences.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seq", "mu", "sigma", "score"])

        for i in range(args.num_batches):
            seqs, _ = sampler.sample(B, L)
            ranked = sampler.score_and_rank(seqs)
            for j in range(seqs.size(0)):
                writer.writerow([
                    "".join(INT_TO_BASE[t] for t in seqs[j].tolist()),
                    f"{ranked['mu'][j].item():.4f}",
                    f"{ranked['sigma'][j].item():.4f}",
                    f"{ranked['score'][j].item():.4f}",
                ])
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()