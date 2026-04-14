"""
Ctrl-DNA baseline runner.

Paper:
    Li et al., "Ctrl-DNA: Controllable Cell-Type-Specific Regulatory DNA
    Design via Constrained RL", 2025.
    https://github.com/xinyuli1204/Ctrl-DNA

Ctrl-DNA fine-tunes a pretrained regLM autoregressive DNA model with
REINFORCE under a Lagrangian constraint. This wrapper assumes a trained
agent checkpoint already exists (produced by the upstream training
scripts — e.g. `reinforce_lagrange_enhancers.sh`) and only runs *sampling*.

What this script does
---------------------
1. Loads a trained Ctrl-DNA regLM agent from disk.
2. Samples N enhancer sequences conditioned on a cell-type label prefix.
3. Scores each sequence with **our** EnsembleRegressor so that
   (mu_hat, sigma_hat) are directly comparable to numbers produced by
   the other baseline runners and scripts/guided_sampler.py.
4. Writes a CSV with columns `seq, mu, sigma, score` where
       score = mu - gamma_rank * sigma.

CLI
---
    python -m scripts.run_ctrl_dna_baseline \
        --config configs/ctrl_dna_baseline.yaml \
        --out_dir ./runs/ctrl_dna_baseline
"""

import argparse
import csv
import os
import sys

import torch
import yaml
from ml_collections.config_dict import ConfigDict
from tqdm import tqdm

from sequence_generation.model.ensemble_regressor import EnsembleRegressor


INT_TO_BASE = {0: "A", 1: "C", 2: "G", 3: "T"}
BASE_TO_INT = {"A": 0, "C": 1, "G": 2, "T": 3}


# ---------------------------------------------------------------------------
# Ctrl-DNA loader
# ---------------------------------------------------------------------------
def load_ctrl_dna_agent(ctrl_cfg, device: torch.device):
    """Import Ctrl-DNA from disk and instantiate the trained regLM agent."""
    repo_path = ctrl_cfg.repo_path
    if not os.path.isdir(repo_path):
        raise FileNotFoundError(
            f"Ctrl-DNA repo not found at {repo_path}. Clone "
            "https://github.com/xinyuli1204/Ctrl-DNA and set ctrl_dna.repo_path."
        )
    sys.path.insert(0, repo_path)
    cwd_backup = os.getcwd()
    os.chdir(repo_path)
    try:
        # Ctrl-DNA wraps regLM's Lightning hyenadna / transformer in a
        # RL-agent module; the class name varies by version (`Agent`,
        # `RLAgent`, `Ctrldna`), so we try known entry points.
        try:
            from agent import Agent as AgentCls  # noqa: WPS433
        except ImportError:
            from ctrl_dna.agent import Agent as AgentCls  # noqa: WPS433

        ckpt_path = ctrl_cfg.checkpoint_path
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Ctrl-DNA agent checkpoint not found at {ckpt_path}. "
                "Train with the upstream repo first "
                "(e.g. reinforce_lagrange_enhancers.sh)."
            )

        # Ctrl-DNA uses PyTorch-Lightning's load_from_checkpoint.
        agent = AgentCls.load_from_checkpoint(ckpt_path, map_location=device)
        agent = agent.to(device).eval()
        for p in agent.parameters():
            p.requires_grad_(False)
        print(f"[Ctrl-DNA] loaded agent checkpoint from {ckpt_path}")
        return agent
    finally:
        os.chdir(cwd_backup)


# ---------------------------------------------------------------------------
# Sampling + scoring
# ---------------------------------------------------------------------------
@torch.no_grad()
def sample_ctrl_dna(agent, ctrl_cfg, num_batches: int,
                    batch_size: int, seq_length: int,
                    device: torch.device) -> torch.Tensor:
    """
    Returns a (N, L) LongTensor of token ids (A=0, C=1, G=2, T=3).

    Ctrl-DNA's Agent follows regLM's `get_data` / `decode` conventions:
    `get_data(label, n, length)` returns raw token indices; `decode`
    turns them into ACGT strings. We go the other way so we can reuse
    our ensemble regressor which expects integer-coded sequences.
    """
    prefix = ctrl_cfg.get("cell_type_prefix", None)
    temperature = float(ctrl_cfg.get("temperature", 1.0))
    top_k = int(ctrl_cfg.get("top_k", 0))
    top_p = float(ctrl_cfg.get("top_p", 1.0))

    all_seqs = []
    for _ in tqdm(range(num_batches), desc="Ctrl-DNA sampling"):
        # Preferred API: agent.get_data returns decoded ACGT strings.
        if hasattr(agent, "get_data"):
            decoded = agent.get_data(
                label=prefix,
                n=batch_size,
                length=seq_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
        elif hasattr(agent, "sample"):
            decoded = agent.sample(
                prefix=prefix,
                n=batch_size,
                length=seq_length,
                temperature=temperature,
            )
        else:
            raise AttributeError(
                "Ctrl-DNA agent has neither `get_data` nor `sample`; "
                "adapt scripts/run_ctrl_dna_baseline.py for this checkpoint."
            )

        # `decoded` may be either a list of ACGT strings or a LongTensor.
        if isinstance(decoded, torch.Tensor):
            ids = decoded.detach().cpu()
        else:
            ids = torch.tensor(
                [[BASE_TO_INT[c] for c in s[:seq_length]] for s in decoded],
                dtype=torch.long,
            )
        all_seqs.append(ids)
    return torch.cat(all_seqs, dim=0)


def load_ensemble(cfg, alphabet_size: int, device: torch.device) -> EnsembleRegressor:
    ens_cfg = cfg.ensemble
    ens = EnsembleRegressor(
        alphabet_size=alphabet_size,
        num_members=ens_cfg.get("num_members", 5),
        hidden_dim=ens_cfg.get("hidden_dim", 128),
        depth=ens_cfg.get("depth", 4),
        dropout=ens_cfg.get("dropout", 0.1),
    )
    ckpt = ens_cfg.checkpoint_path
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Ensemble checkpoint not found at {ckpt}.")
    sd = torch.load(ckpt, map_location="cpu")
    ens.load_state_dict(sd.get("model", sd))
    ens.to(device).eval()
    for p in ens.parameters():
        p.requires_grad_(False)
    print(f"[ensemble] loaded {ckpt}")
    return ens


@torch.no_grad()
def score_with_ensemble(ens: EnsembleRegressor, seqs: torch.Tensor,
                        batch_size: int = 256):
    mus, sigmas = [], []
    for i in range(0, seqs.size(0), batch_size):
        chunk = seqs[i:i + batch_size]
        mu, var = ens.mu_sigma2({"seqs": chunk.to(next(ens.parameters()).device)})
        mus.append(mu.squeeze(-1).cpu())
        sigmas.append(var.clamp_min(1e-12).sqrt().squeeze(-1).cpu())
    return torch.cat(mus, 0), torch.cat(sigmas, 0)


def seqs_to_strings(seqs: torch.Tensor):
    return ["".join(INT_TO_BASE[t] for t in row.tolist()) for row in seqs]


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

    agent = load_ctrl_dna_agent(config.ctrl_dna, device)
    ens = load_ensemble(config, config.model.alphabet_size, device)

    s = config.sampling
    seqs = sample_ctrl_dna(
        agent, config.ctrl_dna,
        num_batches=s.num_batches, batch_size=s.batch_size,
        seq_length=s.seq_length, device=device,
    )
    mu, sigma = score_with_ensemble(ens, seqs)
    gamma = float(s.get("gamma_rank", 1.0))
    score = mu - gamma * sigma

    out_path = os.path.join(args.out_dir, "ctrl_dna_sequences.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seq", "mu", "sigma", "score"])
        for string, m, sg, sc in zip(seqs_to_strings(seqs),
                                     mu.tolist(), sigma.tolist(), score.tolist()):
            writer.writerow([string, f"{m:.4f}", f"{sg:.4f}", f"{sc:.4f}"])
    print(f"[ctrl_dna_baseline] wrote {seqs.size(0)} sequences -> {out_path}")


if __name__ == "__main__":
    main()
