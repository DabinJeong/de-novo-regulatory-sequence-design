"""
SVDD baseline runner.

Paper:
    Li, Zhao, Uehara et al., "Derivative-Free Guidance in Continuous and
    Discrete Diffusion Models with Soft Value-Based Decoding" (SVDD),
    NeurIPS 2024.  https://github.com/masa-ue/SVDD

What this script does
---------------------
1. Loads SVDD's `diffusion_gosai.Diffusion` (MDLM-based masked discrete
   diffusion over enhancer sequences) plus the Gosai activity oracle.
2. Generates enhancer sequences using one of three SVDD decoding modes:
       "mc"      -> `controlled_sample(...)`         (SVDD-MC)
       "tweedie" -> `controlled_sample_tweedie(...)` (SVDD-PM)
       "plain"   -> `_sample(...)`                   (unconditional MDLM)
3. Scores each sequence with our PropertyScorer so that
   (mu_hat, sigma_hat) are directly comparable to numbers produced by
   scripts/guided_sampler.py and the other baseline runners.
4. Writes a CSV with columns `seq, mu, sigma, score` where
       score = mu - gamma_rank * sigma
   matching the header emitted by our guided sampler.

Prerequisites
-------------
- A checkout of https://github.com/masa-ue/SVDD on disk, containing
  `diffusion_gosai.py`, `oracle.py`, `configs_gosai/`, `models/`, etc.
- SVDD's pretrained MDLM Gosai checkpoint (distributed with the repo).
- Our PropertyScorer checkpoint (train via scripts.main_guided --train_ensemble,
  which invokes scripts.ensemble_trainer).

CLI
---
    python -m scripts.run_svdd_baseline \
        --config configs/svdd_baseline.yaml \
        --out_dir ./runs/svdd_baseline
"""

import argparse
import csv
import os
import sys

import torch
import yaml
from ml_collections.config_dict import ConfigDict
from tqdm import tqdm

from sequence_generation.model.property_scorer import PropertyScorer


INT_TO_BASE = {0: "A", 1: "C", 2: "G", 3: "T"}


# ---------------------------------------------------------------------------
# SVDD loader
# ---------------------------------------------------------------------------
def load_svdd_model(svdd_cfg, device: torch.device):
    """Import SVDD from disk and instantiate the Gosai Diffusion model + oracle."""
    repo_path = svdd_cfg.repo_path
    if not os.path.isdir(repo_path):
        raise FileNotFoundError(
            f"SVDD repo not found at {repo_path}. Clone "
            "https://github.com/masa-ue/SVDD and set svdd.repo_path."
        )
    sys.path.insert(0, repo_path)
    cwd_backup = os.getcwd()
    os.chdir(repo_path)
    try:
        from hydra import compose, initialize_config_dir  # noqa: WPS433 (lazy import)
        from hydra.core.global_hydra import GlobalHydra
        # DRAKES' reward_oracle_eval.ckpt (used by SVDD's get_gosai_oracle)
        # predates grelu's data_params-at-top-level format; patch the loader
        # before SVDD's oracle module imports / loads.
        from sequence_generation.utils.grelu_compat import patch_grelu_lightning_compat
        patch_grelu_lightning_compat()
        import diffusion_gosai as diffusion_mod
        import oracle as oracle_mod

        # initialize() resolves config_path relative to the *calling file*, not
        # cwd, so we use initialize_config_dir with an absolute path inside the
        # SVDD checkout.
        cfg_dir = os.path.abspath(
            os.path.join(repo_path,
                         svdd_cfg.get("hydra_config_dir", "configs_gosai"))
        )
        GlobalHydra.instance().clear()
        initialize_config_dir(
            config_dir=cfg_dir,
            job_name="svdd_baseline",
            version_base=None,
        )
        cfg = compose(config_name=svdd_cfg.get("hydra_config_name",
                                               "config_gosai.yaml"))

        ckpt_path = svdd_cfg.checkpoint_path
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"SVDD MDLM checkpoint not found at {ckpt_path}. "
                "Set svdd.checkpoint_path in the config."
            )
        cfg.eval.checkpoint_path = ckpt_path

        # SVDD loads MDLM via Lightning's load_from_checkpoint path.
        model = diffusion_mod.Diffusion.load_from_checkpoint(
            ckpt_path, config=cfg
        ).to(device)
        model.eval()

        # The oracle is used as the reward/value function during decoding.
        # `get_gosai_oracle` returns the pretrained activity predictor from
        # Gosai et al. (same as DRAKES). For MC mode, SVDD uses it split
        # into (embedding, head); for Tweedie/PM mode, SVDD uses the
        # wrapped model end-to-end.
        oracle_model = oracle_mod.get_gosai_oracle().to(device).eval()
        for p in oracle_model.parameters():
            p.requires_grad_(False)

        mode = svdd_cfg.get("mode", "tweedie")
        print(f"[SVDD] loaded MDLM checkpoint from {ckpt_path}; mode={mode}")
        return model, oracle_model, mode
    finally:
        os.chdir(cwd_backup)


def _split_oracle(oracle_model):
    """
    SVDD's `controlled_sample` takes (pre_scorer_embedding, pre_scorer_head).
    The Gosai oracle exposes a `.embedding` and `.head` (or similar) attribute
    pair; if the names differ on a given SVDD checkpoint, the caller can edit
    this helper. We fall back to treating the whole oracle as embedding and a
    no-op head if no split is found.
    """
    emb = getattr(oracle_model, "embedding", None)
    head = getattr(oracle_model, "head", None)
    if emb is not None and head is not None:
        return emb, head
    # Some SVDD checkpoints expose the regression head under different names.
    for emb_attr, head_attr in [
        ("encoder", "regressor"),
        ("trunk", "head"),
        ("backbone", "readout"),
    ]:
        emb = getattr(oracle_model, emb_attr, None)
        head = getattr(oracle_model, head_attr, None)
        if emb is not None and head is not None:
            return emb, head
    raise AttributeError(
        "Could not locate (embedding, head) split on the SVDD oracle. "
        "Please adapt _split_oracle() for your specific checkpoint."
    )


# ---------------------------------------------------------------------------
# Sampling + scoring
# ---------------------------------------------------------------------------
@torch.no_grad()
def sample_svdd(model, oracle_model, mode: str, svdd_cfg,
                num_batches: int, batch_size: int) -> torch.Tensor:
    """Returns a (N, L) LongTensor of token ids."""
    num_steps = svdd_cfg.get("num_steps", None)
    sample_M  = int(svdd_cfg.get("sample_M", 10))
    options   = bool(svdd_cfg.get("options", True))
    task      = svdd_cfg.get("task", "dna")

    samples = []
    for _ in tqdm(range(num_batches), desc=f"SVDD[{mode}] sampling"):
        if mode == "plain":
            out = model._sample(num_steps=num_steps, eval_sp_size=batch_size)
        elif mode == "mc":
            pre_emb, pre_head = _split_oracle(oracle_model)
            out = model.controlled_sample(
                pre_scorer_embedding=pre_emb,
                pre_scorer_head=pre_head,
                num_steps=num_steps,
                eval_sp_size=batch_size,
                sample_M=sample_M,
            )
        elif mode == "tweedie":
            out = model.controlled_sample_tweedie(
                reward_model=oracle_model,
                num_steps=num_steps,
                eval_sp_size=batch_size,
                sample_M=sample_M,
                options=options,
                task=task,
            )
        else:
            raise ValueError(f"Unknown svdd.mode={mode!r}; "
                             "expected 'mc', 'tweedie', or 'plain'.")
        samples.append(out.detach().cpu())
    return torch.cat(samples, dim=0)


def load_scorer(cfg, alphabet_size: int, device: torch.device) -> PropertyScorer:
    ens_cfg = cfg.ensemble
    scorer = PropertyScorer(
        alphabet_size=alphabet_size,
        num_members=ens_cfg.get("num_members", 5),
        hidden_dim=ens_cfg.get("hidden_dim", 128),
        depth=ens_cfg.get("depth", 4),
        dropout=ens_cfg.get("dropout", 0.1),
    )
    ckpt = ens_cfg.checkpoint_path
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"PropertyScorer checkpoint not found at {ckpt}.")
    sd = torch.load(ckpt, map_location="cpu")
    scorer.load_state_dict(sd.get("model", sd))
    scorer.to(device).eval()
    for p in scorer.parameters():
        p.requires_grad_(False)
    print(f"[PropertyScorer] loaded {ckpt}")
    return scorer


@torch.no_grad()
def score_sequences(scorer: PropertyScorer, seqs: torch.Tensor,
                    batch_size: int = 256):
    """Returns (mu, sigma) each shape (N,)."""
    mus, sigmas = [], []
    for i in range(0, seqs.size(0), batch_size):
        chunk = seqs[i:i + batch_size]
        mu, var = scorer.mu_sigma2({"seqs": chunk.to(next(scorer.parameters()).device)})
        mus.append(mu.squeeze(-1).cpu())
        sigmas.append(var.clamp_min(1e-12).sqrt().squeeze(-1).cpu())
    return torch.cat(mus, 0), torch.cat(sigmas, 0)


def seqs_to_strings(seqs: torch.Tensor):
    # SVDD's discrete diffusion uses vocab_size=5 (A,C,G,T,MASK). Any
    # residual MASK tokens (index 4) get dropped as 'N' so downstream
    # scoring doesn't explode on an invalid row.
    out = []
    for row in seqs:
        out.append("".join(INT_TO_BASE.get(int(t), "N") for t in row.tolist()))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = ConfigDict(yaml.safe_load(f))
    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(config.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, oracle_model, mode = load_svdd_model(config.svdd, device)
    scorer = load_scorer(config, config.model.alphabet_size, device)

    s = config.sampling
    seqs = sample_svdd(model, oracle_model, mode, config.svdd,
                       s.num_batches, s.batch_size)

    # SVDD outputs can contain the mask token (vocab index 4) if
    # noise_removal is off; we only feed valid A/C/G/T rows to the scorer.
    valid = (seqs < 4).all(dim=-1)
    dropped = int((~valid).sum().item())
    if dropped:
        print(f"[svdd_baseline] dropping {dropped} sequences with residual MASK tokens")
    seqs = seqs[valid]

    mu, sigma = score_sequences(scorer, seqs)
    gamma = float(s.get("gamma_rank", 1.0))
    score = mu - gamma * sigma

    out_path = os.path.join(args.out_dir, "svdd_sequences.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seq", "mu", "sigma", "score"])
        for string, m, sg, sc in zip(seqs_to_strings(seqs),
                                     mu.tolist(), sigma.tolist(), score.tolist()):
            writer.writerow([string, f"{m:.4f}", f"{sg:.4f}", f"{sc:.4f}"])
    print(f"[svdd_baseline] wrote {seqs.size(0)} sequences -> {out_path}")


if __name__ == "__main__":
    main()
