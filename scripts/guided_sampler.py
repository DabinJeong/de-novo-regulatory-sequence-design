"""
Inference-time guided Dirichlet FM sampler (proposal: slide 14-15).

Implements:

    u*(x_t) = u_t(x_t)                               # frozen Dirichlet FM backbone
            + alpha * grad mu_hat(x_t)               # property guidance
            - beta  * grad sigma2_hat(x_t)           # uncertainty penalty (activity cliff)
            - (1 - sqrt(lambda)) * grad log p_t(x_t) # OOD / novelty knob

The backbone weights are NOT updated. All steering happens at solve time
through the velocity field, exactly as described in the proposal:
"Solver-level steering only — no weight update, no fine-tuning".

Modes:
    lambda = 0  -> in-distribution sampling
    lambda -> 1 -> stronger novelty / OOD push

Ranking score (proposal): score(x) = mu_hat - gamma * sigma_hat + delta * novelty(x)
"""

import os
import csv
import torch
import torch.nn.functional as F

from sequence_generation.utils.flow_utils import (
    DirichletConditionalFlow, expand_simplex,
)
from sequence_generation.utils.train_utils import load_generator
from sequence_generation.model.ensemble_regressor import EnsembleRegressor


INT_TO_BASE = {0: "A", 1: "C", 2: "G", 3: "T"}


class GuidedSampler:
    """Frozen Dirichlet FM backbone + ensemble-guided velocity field."""

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
        ckpt_path = ens_cfg.get("checkpoint_path", None)
        if ckpt_path and os.path.exists(ckpt_path):
            sd = torch.load(ckpt_path, map_location="cpu")
            self.ensemble.load_state_dict(sd.get("model", sd))
            print(f"[GuidedSampler] loaded ensemble checkpoint from {ckpt_path}")
        else:
            print(f"[GuidedSampler] WARNING: no ensemble checkpoint at {ckpt_path}; "
                  f"using randomly initialised property predictor.")
        self.ensemble.to(self.device).eval()

        # ---- guidance hyper-parameters ------------------------------------
        s = config.sampling
        self.alpha_g  = s.get("alpha_guidance", 1.0)   # property weight
        self.beta_g   = s.get("beta_uncertainty", 1.0) # uncertainty penalty
        self.lam      = s.get("ood_lambda", 0.0)       # 0..1 novelty knob
        self.gamma_r  = s.get("gamma_rank", 1.0)       # ranking penalty on sigma
        self.delta_r  = s.get("delta_rank", 0.0)       # ranking bonus on novelty
        self.n_steps  = s.get("n_steps", 128)
        self.flow_temp = s.get("flow_temp", 1.0)
        self.prior_pseudocount = s.get("prior_pseudocount", 0.1)

    # ----------------------------------------------------------------------
    # property + uncertainty gradients in the simplex space
    # ----------------------------------------------------------------------

    def _mu_sigma2_grads(self, xt: torch.Tensor):
        """
        Returns (grad_mu, grad_sigma2, mu, sigma2), each grad of shape (B,L,K).
        xt is a soft simplex point; the ensemble is differentiated through it.
        """
        B, L, K = xt.shape
        x = xt.detach().clone().requires_grad_(True)

        # The SeqRegressor expects a token-id tensor via x['seqs']; we
        # bypass the embedding by feeding the soft mixture directly.
        # ensemble member -> SeqRegressor: we re-use the embedder weights as
        # a (A, hidden) matrix and form the soft embedding x @ W.
        preds = []
        for m in self.ensemble.members:
            W = m.embedder.weight                      # (A, H)
            soft_emb = torch.einsum("blk,kh->blh", x, W)
            h = soft_emb.transpose(1, 2)
            xs = [soft_emb]
            for conv in m.convs:
                h = torch.tanh(conv(h))
                xs.append(h.transpose(1, 2))
            feat = torch.cat(xs, dim=-1)
            gate = m.sigmoid_linear(feat) * m.tanh_linear(feat)
            pooled = torch.tanh(gate.mean(dim=1))
            preds.append(m.final_linear(pooled))       # (B,1)
        preds = torch.stack(preds, dim=0)              # (K_ens, B, 1)
        mu = preds.mean(0)                             # (B,1)
        var = preds.var(0, unbiased=False)             # (B,1)

        grad_mu = torch.autograd.grad(mu.sum(), x, retain_graph=True)[0]
        grad_var = torch.autograd.grad(var.sum(), x, retain_graph=False)[0]

        # zero-mean over the simplex axis (consistent with classifier guidance)
        grad_mu  = grad_mu  - grad_mu.mean(-1, keepdim=True)
        grad_var = grad_var - grad_var.mean(-1, keepdim=True)
        return grad_mu.detach(), grad_var.detach(), mu.detach(), var.detach()

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
            u_t = (flow_probs.unsqueeze(-2) * cond_flows).sum(-1)   # (B,L,K)

            # ---- ensemble guidance terms ---------------------------------
            grad_mu, grad_var, _, _ = self._mu_sigma2_grads(xt)

            # ---- novelty / OOD term: -(1 - sqrt(lambda)) * grad log p_t ---
            # The Dirichlet FM "score" can be approximated from u_t under the
            # conditional flow factorisation; we use grad log p_t ~ u_t / g_t
            # with g_t absorbed into the lambda knob (matches MOOD / proposal).
            novelty_term = -(1.0 - self.lam ** 0.5) * u_t

            u_star = (
                u_t
                + self.alpha_g * grad_mu
                - self.beta_g  * grad_var
                + novelty_term
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
        """
        seqs: (B, L) token ids
        ref_seqs: (N, L) optional in-distribution reference set for novelty
                  (Hamming distance to nearest neighbour, normalised by L).
        Returns dict(score, mu, sigma, novelty), each (B,).
        """
        mu, var = self.ensemble.mu_sigma2({"seqs": seqs.to(self.device)})
        mu = mu.squeeze(-1)
        sigma = var.clamp_min(1e-12).sqrt().squeeze(-1)

        if ref_seqs is not None and ref_seqs.numel() > 0:
            r = ref_seqs.to(self.device)
            # min Hamming distance to reference, normalised
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
