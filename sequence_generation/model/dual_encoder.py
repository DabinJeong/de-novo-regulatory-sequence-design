"""
Dual encoder for semantic / variation disentanglement (proposal: slide 14-15).

Architecture
------------
A shared CNN body encodes a (noisy) sequence into two latents:

    s = semantic_encoder(x)   -- functional motif representation
                                 (TF binding, attribution maps)
                                 -> invariant across environments
    v = variation_encoder(x)  -- background composition
                                 (GC content, k-mer context)
                                 -> diverse across environments

A decoder reconstructs the sequence from (s, v).

Training objective (slide 15)
-----------------------------
    L = L_recon  +  lambda_inv * L_inv(s)  +  lambda_div * L_div(v)

  L_recon : cross-entropy reconstruction of x from (s, v)
  L_inv(s): semantic should be invariant across environments
            -> penalise per-environment mean drift of s
  L_div(v): variation should *carry* environment info
            -> small env classifier on v; minimise its CE
            (a successful env classifier means v captures background shift)

Environments come from `task_id` in the Gosai dataset (one task per cell line)
but any categorical environment label works.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from sequence_generation.model.denoising_classifier import CNNModel


# ---------------------------------------------------------------------------
# Encoder body (shared CNN, two projection heads)
# ---------------------------------------------------------------------------

class DualEncoder(nn.Module):
    def __init__(self, args, alphabet_size: int, sem_dim: int, var_dim: int):
        super().__init__()
        self.sem_dim = sem_dim
        self.var_dim = var_dim

        self._cnn = CNNModel(args, alphabet_size, num_cls=1, classifier=False)
        hidden = args.hidden_dim
        self.pool = nn.AdaptiveAvgPool1d(1)

        # two separate projection heads -> encourages decoupled latents
        self.head_s = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, sem_dim),
        )
        self.head_v = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, var_dim),
        )

    def _encode_hidden(self, x, t):
        cnn = self._cnn
        time_emb = F.relu(cnn.time_embedder(t))
        feat = F.relu(cnn.linear(x.permute(0, 2, 1)))
        for i in range(cnn.num_layers):
            h = cnn.dropout(feat.clone())
            h = h + cnn.time_layers[i](time_emb)[:, :, None]
            h = cnn.norms[i](h.permute(0, 2, 1))
            h = F.relu(cnn.convs[i](h.permute(0, 2, 1)))
            feat = h + feat if h.shape == feat.shape else h
        return feat                                              # (B, hidden, L)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        feat = self._encode_hidden(x, t)
        pooled = self.pool(feat).squeeze(-1)                     # (B, hidden)
        s = self.head_s(pooled)                                  # (B, sem_dim)
        v = self.head_v(pooled)                                  # (B, var_dim)
        return s, v


# ---------------------------------------------------------------------------
# Decoder: (s, v) -> per-position logits
# ---------------------------------------------------------------------------

class SequenceDecoder(nn.Module):
    def __init__(self, sem_dim: int, var_dim: int, hidden_dim: int,
                 seq_len: int, alphabet_size: int):
        super().__init__()
        self.seq_len = seq_len
        self.expand = nn.Linear(sem_dim + var_dim, hidden_dim * seq_len)
        self.refine = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 9, padding=4), nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 9, padding=4), nn.ReLU(),
            nn.Conv1d(hidden_dim, alphabet_size, 1),
        )

    def forward(self, s: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        B = s.size(0)
        h = self.expand(torch.cat([s, v], -1)).view(B, -1, self.seq_len)
        return self.refine(h).permute(0, 2, 1)                   # (B, L, K)


# ---------------------------------------------------------------------------
# Auxiliary environment classifier on v (drives L_div)
# ---------------------------------------------------------------------------

class EnvClassifier(nn.Module):
    def __init__(self, var_dim: int, hidden_dim: int, num_envs: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(var_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_envs),
        )

    def forward(self, v):
        return self.net(v)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def l_inv_semantic(s: torch.Tensor, env_ids: torch.Tensor, num_envs: int) -> torch.Tensor:
    """
    Semantic invariance: per-environment mean of s should not drift.
    Loss = sum_e || mean_e(s) - mean(s) ||^2 / num_envs
    """
    global_mean = s.mean(0, keepdim=True)
    losses = []
    for e in range(num_envs):
        mask = env_ids == e
        if mask.sum() > 0:
            env_mean = s[mask].mean(0, keepdim=True)
            losses.append(((env_mean - global_mean) ** 2).sum())
    if not losses:
        return s.new_zeros(())
    return torch.stack(losses).mean()


def l_div_variation(env_logits: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
    """
    Variation diversification: v should *carry* environment information,
    so a small classifier on v should be able to recover env_id.
    Minimising CE here means v becomes env-discriminative (== diverse across envs).
    """
    return F.cross_entropy(env_logits, env_ids)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class DualEncoderModel(nn.Module):
    """Encoder + Decoder + EnvClassifier with the L_recon + L_inv + L_div objective."""

    def __init__(self, config, num_envs: int):
        super().__init__()
        args = config.model
        self.alphabet_size = args.alphabet_size
        self.seq_len       = config.dataset.seq_length
        self.num_envs      = num_envs

        de_cfg = config.get("dual_encoder", {})
        self.sem_dim    = de_cfg.get("sem_dim", 64)
        self.var_dim    = de_cfg.get("var_dim", 64)
        self.lambda_inv = de_cfg.get("lambda_inv", 1.0)
        self.lambda_div = de_cfg.get("lambda_div", 1.0)
        self.alpha_max  = args.alpha_max

        self.encoder = DualEncoder(args, self.alphabet_size, self.sem_dim, self.var_dim)
        self.decoder = SequenceDecoder(self.sem_dim, self.var_dim,
                                       args.hidden_dim, self.seq_len, self.alphabet_size)
        self.env_clf = EnvClassifier(self.var_dim, args.hidden_dim, num_envs)

    # ------------------------------------------------------------------
    def compute_loss(self, x: torch.Tensor, env_ids: torch.Tensor,
                     t: Optional[torch.Tensor] = None) -> dict:
        """
        x       : (B, L)  token ids
        env_ids : (B,)    environment / task labels in [0, num_envs)
        t       : (B,)    optional Dirichlet time; if None, sampled from [1, alpha_max]
        """
        B, L = x.shape
        K = self.alphabet_size
        device = x.device

        if t is None:
            t = 1.0 + torch.rand(B, device=device) * (self.alpha_max - 1.0)

        # Build a noisy simplex input as in Dirichlet FM training.
        x_onehot = F.one_hot(x, K).float()
        alphas_ = torch.ones(B, L, K, device=device) + x_onehot * (t[:, None, None] - 1)
        x_noisy = torch.distributions.Dirichlet(alphas_).sample()

        s, v = self.encoder(x_noisy, t)

        # 1. reconstruction
        logits = self.decoder(s, v)
        recon = F.cross_entropy(logits.reshape(B * L, K), x.reshape(B * L))

        # 2. semantic invariance on s
        l_inv = l_inv_semantic(s, env_ids, self.num_envs)

        # 3. variation diversification on v
        env_logits = self.env_clf(v)
        l_div = l_div_variation(env_logits, env_ids)

        total = recon + self.lambda_inv * l_inv + self.lambda_div * l_div
        return {"loss": total, "recon": recon, "inv": l_inv, "div": l_div}

    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode(self, x: torch.Tensor, t: Optional[torch.Tensor] = None):
        """Inference helper: returns (s, v) for token-id input."""
        B = x.size(0)
        device = x.device
        if t is None:
            t = torch.full((B,), float(self.alpha_max), device=device)
        x_onehot = F.one_hot(x, self.alphabet_size).float()
        return self.encoder(x_onehot, t)
