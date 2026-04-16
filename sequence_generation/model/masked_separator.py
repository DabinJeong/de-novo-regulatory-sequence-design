"""
GIL-style masked sub-sequence separator for MPRA sequences.

Adapted from:
    Li et al., "Learning Invariant Graph Representations for Out-of-Distribution
    Generalization" (GIL), NeurIPS 2022
    https://openreview.net/forum?id=acKK8MQe2xc

GIL's three modules, translated to sequences
--------------------------------------------
(1) Invariant subgraph generator -> position-wise mask separator
        M in [0, 1]^L splits x into x_st (stable / functional) and
        x_en (environmental / background).

(2) Latent environment inference on the variant side
        Embed x_en, run K-means on those embeddings, use the cluster
        assignment as inferred environment label ê.

(3) Invariant learning module
        V-REx penalty on per-env MSE risks using only x_st -> y_head:
            L_inv = Var_e [ E_{(x,y) in ê=e} MSE(y_head(x_st), y) ]

Training objective
------------------
    L = L_sta + lambda_reg * L_reg + beta_inv * L_inv

    L_sta : x_st alone must predict y                    (sufficiency)
    L_reg : mask ratio close to learnable rho, bimodal   (non-triviality)
    L_inv : V-REx variance across inferred environments  (invariance)

This module is a standalone V-REx model operating on clean one-hot
sequences. It is independent of the Dirichlet-FM generative backbone
and has no time conditioning or expanded-simplex input.
Modules (1) and (3) live in this file. Module (2) — K-means env inference —
lives in the trainer because it needs a batch-global view and is alternated
with separator updates (standard GIL alternating optimisation).
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared CNN backbone producing per-position features (B, hidden, L)
# ---------------------------------------------------------------------------
class _CNNBackbone(nn.Module):
    """
    Plain 1D CNN backbone (no time conditioning, no expanded simplex).
    Input:  (B, L, K) soft simplex or one-hot
    Output: (B, hidden, L) per-position features
    """

    def __init__(self, args, alphabet_size: int):
        super().__init__()
        hidden = args.hidden_dim
        dropout = getattr(args, "dropout", 0.1)
        num_layers = 5 * getattr(args, "num_cnn_stacks", 1)

        self.in_conv = nn.Conv1d(alphabet_size, hidden, kernel_size=9, padding=4)

        dilations = [1, 1, 4, 16, 64]
        paddings  = [4, 4, 16, 64, 256]
        stacks = getattr(args, "num_cnn_stacks", 1)
        convs = []
        for _ in range(stacks):
            for d, p in zip(dilations, paddings):
                convs.append(nn.Conv1d(hidden, hidden, kernel_size=9,
                                       dilation=d, padding=p))
        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = F.relu(self.in_conv(x.permute(0, 2, 1)))        # (B, H, L)
        for conv, norm in zip(self.convs, self.norms):
            h = self.dropout(feat.clone())
            h = norm(h.permute(0, 2, 1)).permute(0, 2, 1)
            h = F.relu(conv(h))
            feat = h + feat if h.shape == feat.shape else h
        return feat                                           # (B, H, L)


# ---------------------------------------------------------------------------
# Mask-based separator: position-wise M in [0, 1]
# ---------------------------------------------------------------------------
class StableMaskSeparator(nn.Module):
    """
    Predicts a per-position soft mask M in [0,1]^L.
    M_i -> 1 : position i belongs to x_st (stable / functional)
    M_i -> 0 : position i belongs to x_en (environmental / background)
    """

    def __init__(self, args, alphabet_size: int, init_rho: float = 0.3):
        super().__init__()
        self.backbone = _CNNBackbone(args, alphabet_size)
        hidden = args.hidden_dim
        self.mask_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        # Initialise the output-layer bias so sigmoid(bias) = init_rho, i.e.
        # the mask starts at the target stable ratio rather than 0.5. Keeps
        # the system away from the bimodal-term valleys (at M=0 and M=1)
        # that previously swallowed training in the first epoch.
        init_rho = float(min(max(init_rho, 1e-4), 1 - 1e-4))
        with torch.no_grad():
            self.mask_head[-1].bias.fill_(math.log(init_rho / (1.0 - init_rho)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)                              # (B, H, L)
        feat = feat.permute(0, 2, 1)                         # (B, L, H)
        logits = self.mask_head(feat).squeeze(-1)            # (B, L)
        return torch.sigmoid(logits)                         # (B, L)


# ---------------------------------------------------------------------------
# y-head: reads a (sub-)sequence and predicts activity
# ---------------------------------------------------------------------------
class YHead(nn.Module):
    """Small CNN+pool regressor on top of its own backbone."""

    def __init__(self, args, alphabet_size: int, out_dim: int = 1):
        super().__init__()
        self.backbone = _CNNBackbone(args, alphabet_size)
        hidden = args.hidden_dim
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.readout = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)                              # (B, H, L)
        pooled = self.pool(feat).squeeze(-1)                 # (B, H)
        return self.readout(pooled)                          # (B, out_dim)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def mask_regularisation(
    M: torch.Tensor,
    rho: torch.Tensor,
    lambda_bimodal: float = 0.0,
) -> torch.Tensor:
    """
    Encourages:
      (1) average mask value tracks the target rho from both sides
          (symmetric quadratic) — target-expected-L0 anchoring in the
          spirit of Louizos et al. 2018 and Shazeer et al. 2017.
      (2) optional bimodality via M*(1-M), weighted by lambda_bimodal.
          Disabled by default because at lambda_bimodal > 0 the valleys
          at M=0 and M=1 act as attractors: combined with the mean term
          preferring the side of rho closer to {0,1}, the separator
          collapses onto one of those degenerate solutions in the first
          epoch. Turn on only after a soft mask has stabilised around
          rho (e.g. as a second-stage annealing).
    """
    mean_term = (M.mean() - rho) ** 2
    if lambda_bimodal > 0.0:
        bimodal_term = (M * (1.0 - M)).mean()
        return mean_term + lambda_bimodal * bimodal_term
    return mean_term


def v_rex_penalty(
    per_sample_loss: torch.Tensor,
    env_ids: torch.Tensor,
    num_envs: int,
    min_samples_per_env: int = 2,
) -> torch.Tensor:
    """
    V-REx penalty = Var_e [ mean_{i in e}(per_sample_loss_i) ]

    per_sample_loss : (B,)  element-wise loss, e.g. squared error for each sample
    env_ids         : (B,)  inferred environment ids in [0, num_envs)
    num_envs        : K     total number of inferred environments
    min_samples_per_env :   envs with fewer than this many samples are skipped

    If fewer than 2 envs are populated, returns a zero tensor (penalty undefined).

    Reference:
        Krueger et al., "Out-of-Distribution Generalization via Risk
        Extrapolation (REx)", ICML 2021.
    """
    env_risks = []
    for e in range(num_envs):
        mask = env_ids == e
        if mask.sum() >= min_samples_per_env:
            env_risks.append(per_sample_loss[mask].mean())
    if len(env_risks) < 2:
        return per_sample_loss.new_zeros(())
    env_risks = torch.stack(env_risks)                       # (K_valid,)
    return env_risks.var(unbiased=False)


# ---------------------------------------------------------------------------
# Full model: mask separator + y-head + V-REx invariance
# ---------------------------------------------------------------------------
class MaskedSeparatorModel(nn.Module):
    """
    GIL-style masked sub-sequence separator for MPRA sequences.

    Forward flow per training step (see `compute_loss`):
        1. embed clean one-hot x
        2. predict mask M = separator(x)
        3. build x_st = x * M,  x_en = x * (1 - M)
        4. per_sample = (y_head(x_st) - y)^2   -- element-wise MSE
        5. L_sta = mean(per_sample)
        6. L_inv = v_rex_penalty(per_sample, env_ids)   (if env_ids given)
        7. L_reg = mask_regularisation(M, rho)

    env_ids are passed in from the trainer (K-means on embed_en(x_en) at
    the start of each epoch). When env_ids is None (e.g. warmup epoch
    before the first K-means run), L_inv is skipped.
    """

    def __init__(self, config):
        super().__init__()
        args = config.model
        self.alphabet_size = args.alphabet_size
        self.seq_len       = config.dataset.seq_length

        de_cfg = config.get("masked_separator", config.get("dual_encoder", {}))
        self.beta_inv       = de_cfg.get("beta_inv", 1.0)         # V-REx weight
        self.lambda_reg     = de_cfg.get("lambda_reg", 1.0)       # overall L_reg scale
        self.lambda_bimodal = de_cfg.get("lambda_bimodal", 0.0)   # bimodal sub-weight
        self.num_envs       = de_cfg.get("num_envs", 3)           # K for K-means
        self.target_idx     = de_cfg.get("target_idx", None)      # None -> mean over tasks
        init_rho            = de_cfg.get("init_rho", 0.3)

        self.separator = StableMaskSeparator(args, self.alphabet_size, init_rho=init_rho)
        self.y_head    = YHead(args, self.alphabet_size, out_dim=1)

        # Dedicated encoder for x_en used by K-means env inference and the
        # sampler's env-push guidance. Kept separate from y_head.backbone so
        # that collapsing the mask (x_st -> 0, y_head tuned for "0 -> const")
        # does not corrupt the env embedding space. This backbone receives no
        # gradient from L_sta; it acts as a frozen-at-init random projection
        # on x_en, which is enough for K-means clustering but breaks the
        # self-reinforcing mask=0 collapse loop.
        self.en_encoder = _CNNBackbone(args, self.alphabet_size)

        # Learnable mask-token embedding: positions that get masked out are
        # replaced with this vector rather than zeroed. Keeps the y_head's
        # input at a constant nucleotide-scale magnitude so the degenerate
        # "zero input -> constant output" shortcut is no longer available.
        # This matches the BERT/MLM convention (Devlin et al. 2019) and the
        # rationalization-literature fix for extractor-predictor collapse
        # (Yu et al. 2019 "Rethinking Cooperative Rationalization";
        #  Bastings et al. 2019 warn explicitly that zero-masked positions
        #  leak mask identity to the downstream predictor).
        # Initialised to a uniform distribution over the alphabet.
        self.mask_embed = nn.Parameter(
            torch.ones(self.alphabet_size) / float(self.alphabet_size)
        )

        # Fixed stable ratio target. Previously this was an nn.Parameter,
        # but the gradient dL_reg/drho = -2*(M.mean - rho) pulled rho toward
        # M.mean whenever the mask collapsed, which hollowed out the L_reg
        # penalty over training instead of fighting the collapse. Registered
        # as a buffer so it still moves with .to(device) but receives no
        # gradient.
        self.register_buffer("rho", torch.tensor(float(init_rho)))

    # ------------------------------------------------------------------
    def _target(self, clss: torch.Tensor) -> torch.Tensor:
        """
        clss: (B, num_tasks) activity across cell lines
        Returns: (B, 1) regression target
        """
        if self.target_idx is None:
            y = clss.float().mean(dim=-1, keepdim=True)
        else:
            y = clss[:, self.target_idx:self.target_idx + 1].float()
        return y

    # ------------------------------------------------------------------
    def _one_hot(self, x_clean: torch.Tensor) -> torch.Tensor:
        return F.one_hot(x_clean, self.alphabet_size).float()

    # ------------------------------------------------------------------
    def compute_loss(
        self,
        x_clean: torch.Tensor,              # (B, L) token ids
        clss: torch.Tensor,                 # (B, num_tasks)
        env_ids: Optional[torch.Tensor] = None,   # (B,) inferred env ids
    ) -> dict:
        device = x_clean.device

        # 1. clean one-hot input
        x = self._one_hot(x_clean)                           # (B, L, K)

        # 2. separator produces mask
        M = self.separator(x)                                # (B, L)
        M3 = M.unsqueeze(-1)                                 # (B, L, 1)

        # 3. stable / environmental sub-sequences. Positions that are not
        # selected into the sub-sequence get the learnable mask token rather
        # than a zero vector — otherwise y_head can (and does) collapse into
        # "zero input -> constant output", trivialising the mask.
        bkg = self.mask_embed.view(1, 1, -1)                 # (1, 1, K)
        x_st = x * M3 + bkg * (1.0 - M3)
        x_en = x * (1.0 - M3) + bkg * M3

        # 4. per-sample squared error on x_st
        y = self._target(clss)                               # (B, 1)
        y_hat_st = self.y_head(x_st)                         # (B, 1)
        per_sample = ((y_hat_st - y) ** 2).squeeze(-1)       # (B,)

        # 5. L_sta = average MSE  (the "ERM" term inside V-REx)
        L_sta = per_sample.mean()

        # 6. L_inv = V-REx variance across inferred envs
        if env_ids is not None:
            L_inv = v_rex_penalty(per_sample, env_ids.to(device), self.num_envs)
        else:
            L_inv = per_sample.new_zeros(())

        # 7. mask regularisation (rho is a frozen buffer)
        L_reg = mask_regularisation(M, self.rho, lambda_bimodal=self.lambda_bimodal)

        total = L_sta + self.lambda_reg * L_reg + self.beta_inv * L_inv

        return {
            "loss":  total,
            "L_sta": L_sta.detach(),
            "L_reg": L_reg.detach(),
            "L_inv": L_inv.detach(),
            "rho":   self.rho.detach(),
            "mask_mean": M.mean().detach(),
            # expose intermediates if the trainer wants them
            "M":    M.detach(),
            "x_st": x_st.detach(),
            "x_en": x_en.detach(),
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def separate(
        self,
        x: torch.Tensor,
        soft_input: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Inference helper. Returns (M, x_st, x_en).

        x           : (B, L) token ids  if soft_input is False
                      (B, L, K) simplex  if soft_input is True
        soft_input  : format flag for x
        """
        x_soft = x if soft_input else self._one_hot(x)

        M = self.separator(x_soft)                           # (B, L)
        M3 = M.unsqueeze(-1)
        bkg = self.mask_embed.view(1, 1, -1)                 # (1, 1, K)
        x_st = x_soft * M3 + bkg * (1.0 - M3)
        x_en = x_soft * (1.0 - M3) + bkg * M3
        return M, x_st, x_en

    # ------------------------------------------------------------------
    def predict_y_from_st(self, x_st: torch.Tensor) -> torch.Tensor:
        """Differentiable y prediction from stable sub-sequence (used by sampler)."""
        return self.y_head(x_st)

    # ------------------------------------------------------------------
    def embed_en(self, x_en: torch.Tensor) -> torch.Tensor:
        """
        Pooled representation of x_en using a dedicated env encoder that is
        *not* shared with y_head. Used by
            (i)  K-means env inference in the trainer,
            (ii) variation-push guidance in the sampler.
        """
        feat = self.en_encoder(x_en)                         # (B, H, L)
        return feat.mean(dim=-1)                             # (B, H)
