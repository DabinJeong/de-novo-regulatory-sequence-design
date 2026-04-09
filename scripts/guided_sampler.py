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

`env_centroids` are loaded from the dual-encoder checkpoint (the trainer
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
from sequence_generation.model.ensemble_regressor import EnsembleRegressor
from sequence_generation.model.dual_encoder import DualEncoderModel


INT_TO_BASE = {0: "A", 1: "C", 2: "G", 3: "T"}


class GuidedSampler:
    """Frozen Dirichlet FM backbone + ensemble + GIL separator guidance."""

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

        # ---- ensemble property predictor (mu_hat, sigma2_hat) -------------
        ens_cfg = config.get("ensemble", {})
        self.ensemble = EnsembleRegressor(
            alphabet_size=self.K,
            num_members=ens_cfg.get("num_members", 5),
            hidden_dim=ens_cfg.get("hidden_dim", 128),
            depth=ens_cfg.get("depth", 4),
            dropout=ens_cfg.get("dropout", 0.1),
        )
        ens_ckpt = ens_cfg.get("checkpoint_path", None)
        if ens_ckpt and os.path.exists(ens_ckpt):
            sd = torch.load(ens_ckpt, map_location="cpu")
            self.ensemble.load_state_dict(sd.get("model", sd))
            print(f"[GuidedSampler] loaded ensemble checkpoint from {ens_ckpt}")
        else:
            raise FileNotFoundError(
                f"Ensemble checkpoint required but not found at {ens_ckpt}. "
                "Train with `scripts.main_guided --train_ensemble` first."
            )
        self.ensemble.to(self.device).eval()
        for p in self.ensemble.parameters():
            p.requires_grad_(False)

        # ---- GIL-style separator ------------------------------------------
        self.dual_encoder = DualEncoderModel(config)
        de_cfg = config.get("dual_encoder", {})
        de_ckpt = de_cfg.get("checkpoint_path", None)
        if de_ckpt and os.path.exists(de_ckpt):
            ckpt = torch.load(de_ckpt, map_location="cpu")
            self.dual_encoder.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
            print(f"[GuidedSampler] loaded dual-encoder checkpoint from {de_ckpt}")
            # Try to load env centroids saved by the trainer
            env_centroids = ckpt.get("env_centroids", None) if isinstance(ckpt, dict) else None
        else:
            raise FileNotFoundError(
                f"Dual-encoder checkpoint required but not found at {de_ckpt}. "
                "Train with `scripts.dual_encoder_trainer` first."
            )
        self.dual_encoder.to(self.device).eval()
        for p in self.dual_encoder.parameters():
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

        # ---- env centroids: either from checkpoint or recompute ----------
        if env_centroids is not None:
            self.env_centroids = env_centroids.to(self.device)
            print(f"[GuidedSampler] loaded {self.env_centroids.size(0)} env centroids "
                  f"from dual-encoder checkpoint (dim={self.env_centroids.size(-1)})")
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
        alpha_max = float(self.config.model.alpha_max)
        for batch in train_loader:
            x = batch["seqs"].to(self.device)
            B = x.size(0)
            t = torch.full((B,), alpha_max, device=self.device)
            _, _, x_en = self.dual_encoder.separate(x, t=t, soft_input=False)
            h_en = self.dual_encoder.embed_en(x_en, t)        # (B, H)
            feats.append(h_en)
            collected += B
            if collected >= n_samples:
                break
        feats = torch.cat(feats, dim=0)
        centroid = feats.mean(dim=0, keepdim=True)           # (1, H)
        return centroid.detach()

    # ----------------------------------------------------------------------
    # ensemble gradients (property + uncertainty)
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
        for m in self.ensemble.members:
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
    def _gil_grads(self, xt: torch.Tensor, t_batch: torch.Tensor):
        """
        Returns (grad_stable, grad_en_push), each (B, L, K).

        grad_stable  : direction increasing y_head(x_st, t)
        grad_en_push : direction increasing the min-distance from h_en(x_en, t)
                       to the set of training env centroids
        """
        x = xt.detach().clone().requires_grad_(True)

        M = self.dual_encoder.separator(x, t_batch)              # (B, L)
        M3 = M.unsqueeze(-1)
        x_st = x * M3
        x_en = x * (1.0 - M3)

        # (1) stable-sufficiency
        y_st = self.dual_encoder.predict_y_from_st(x_st, t_batch)  # (B, 1)
        grad_stable = torch.autograd.grad(
            y_st.sum(), x, retain_graph=True,
        )[0]

        # (2) env push: distance to NEAREST env centroid
        h_en = self.dual_encoder.embed_en(x_en, t_batch)           # (B, H)
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

            # ---- ensemble guidance ----------------------------------------
            grad_mu, grad_var, _, _ = self._mu_sigma2_grads(xt)

            # ---- GIL separator guidance -----------------------------------
            t_batch = s[None].expand(B)
            grad_stable, grad_en_push = self._gil_grads(xt, t_batch)

            u_star = (
                u_t
                + self.alpha_g * grad_mu
                - self.beta_g  * grad_var
                + self.eta_st  * grad_stable
                + self.xi_en * (self.lam ** 0.5) * grad_en_push
            )

            xt = (xt + u_star * (t - s)).clamp(min=1e-8)
            xt = xt / xt.sum(-1, keepdim=True)

        seq_pred = xt.argmax(-1)
        return seq_pred, xt

    # ----------------------------------------------------------------------
    # ranking: score(x) = mu_hat - gamma * sigma_hat + delta * novelty(x)
    # ----------------------------------------------------------------------
    @torch.no_grad()
    def score_and_rank(self, seqs: torch.Tensor, ref_seqs: torch.Tensor = None):
        mu, var = self.ensemble.mu_sigma2({"seqs": seqs.to(self.device)})
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