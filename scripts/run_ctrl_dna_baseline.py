"""
regLM baseline runner (option-(b) variant of the Ctrl-DNA comparison).

This script is named after Ctrl-DNA (Li et al., 2025) because it lives in
the Ctrl-DNA repository and uses Ctrl-DNA's vendored copy of regLM.
However, it does NOT run Ctrl-DNA's constrained-RL fine-tune — that
would require a trained agent checkpoint we do not have. Instead, we
sample from the pretrained regLM model (Lal et al., "regLM: Designing
Realistic Regulatory DNA with Autoregressive Language Models", 2024)
conditioned on a cell-type label prefix. This is the "base policy" that
Ctrl-DNA starts from; reporting it alongside Ctrl-DNA would isolate the
contribution of the RL fine-tune, but in our setup it stands in for
Ctrl-DNA itself as the closest faithful baseline we can run.

What this script does
---------------------
1. Loads a pretrained regLM `LightningModel` checkpoint via Lightning's
   `load_from_checkpoint`, with `<ctrl_dna_repo>/ctrl_dna` added to
   sys.path so `src.reglm.*` imports resolve.
2. Generates enhancer sequences with `model.generate(labels=[prefix]*N,
   max_new_tokens=L, temperature, top_k, top_p, seed)`. regLM's label
   convention for the Gosai enhancer task is a 3-char binary string:
       "100" -> HepG2,  "010" -> K562,  "001" -> SK-N-SH.
3. Scores each sequence with our PropertyScorer so that (mu_hat,
   sigma_hat) are directly comparable to guided_sampler / DRAKES / SVDD.
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

from sequence_generation.model.property_scorer import PropertyScorer


INT_TO_BASE = {0: "A", 1: "C", 2: "G", 3: "T"}
BASE_TO_INT = {"A": 0, "C": 1, "G": 2, "T": 3}


# ---------------------------------------------------------------------------
# regLM loader (via Ctrl-DNA's vendored copy)
# ---------------------------------------------------------------------------
def load_reglm_model(ctrl_cfg, device: torch.device):
    """Import regLM from the Ctrl-DNA checkout and load a LightningModel ckpt."""
    repo_path = ctrl_cfg.repo_path
    if not os.path.isdir(repo_path):
        raise FileNotFoundError(
            f"Ctrl-DNA repo not found at {repo_path}. Clone "
            "https://github.com/xinyuli1204/Ctrl-DNA and set ctrl_dna.repo_path."
        )
    # regLM code lives at <repo>/ctrl_dna/src/reglm; `src.reglm.*` imports
    # resolve once <repo>/ctrl_dna is on sys.path.
    inner = os.path.join(repo_path, "ctrl_dna")
    for p in (inner, repo_path):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)

    from src.reglm.lightning import LightningModel  # noqa: WPS433 (lazy import)

    ckpt_path = ctrl_cfg.checkpoint_path
    if not ckpt_path or not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"regLM checkpoint not found at {ckpt_path!r}. Point "
            "ctrl_dna.checkpoint_path at a pretrained regLM LightningModel "
            "checkpoint (Lal et al. 2024). An empty path would give random "
            "label embeddings and no meaningful cell-type conditioning."
        )
    model = LightningModel.load_from_checkpoint(ckpt_path, map_location=device)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    print(f"[regLM] loaded LightningModel checkpoint from {ckpt_path}; "
          f"label_len={getattr(model, 'label_len', '?')}, "
          f"seq_len={getattr(model, 'seq_len', '?')}")
    return model


# ---------------------------------------------------------------------------
# Sampling + scoring
# ---------------------------------------------------------------------------
@torch.no_grad()
def sample_reglm(model, ctrl_cfg, num_batches: int,
                 batch_size: int, seq_length: int,
                 seed: int) -> torch.Tensor:
    """Returns a (N, L) LongTensor of token ids (A=0, C=1, G=2, T=3)."""
    prefix = ctrl_cfg.get("cell_type_prefix", None)
    if prefix is None:
        raise ValueError("ctrl_dna.cell_type_prefix must be set "
                         "(e.g. '100' for HepG2, '010' for K562, '001' for SK-N-SH).")
    label_len = getattr(model, "label_len", None)
    if label_len is not None and len(prefix) != label_len:
        raise ValueError(
            f"ctrl_dna.cell_type_prefix={prefix!r} has length {len(prefix)}, "
            f"but the loaded regLM expects label_len={label_len}."
        )

    temperature = float(ctrl_cfg.get("temperature", 1.0))
    top_k = int(ctrl_cfg.get("top_k", 0)) or None
    top_p = float(ctrl_cfg.get("top_p", 1.0))
    if top_p >= 1.0:
        top_p = None

    all_seqs = []
    for b in tqdm(range(num_batches), desc="regLM sampling"):
        decoded = model.generate(
            labels=[prefix] * batch_size,
            max_new_tokens=seq_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed + b,
        )
        rows = []
        for s in decoded:
            s = s[:seq_length].upper()
            ids = [BASE_TO_INT.get(c, -1) for c in s]
            if len(ids) < seq_length or any(i < 0 for i in ids):
                continue  # drop non-ACGT or short outputs
            rows.append(ids)
        if not rows:
            continue
        all_seqs.append(torch.tensor(rows, dtype=torch.long))
    if not all_seqs:
        raise RuntimeError("regLM produced no valid ACGT sequences.")
    return torch.cat(all_seqs, dim=0)


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
    mus, sigmas = [], []
    for i in range(0, seqs.size(0), batch_size):
        chunk = seqs[i:i + batch_size]
        mu, var = scorer.mu_sigma2({"seqs": chunk.to(next(scorer.parameters()).device)})
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

    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_reglm_model(config.ctrl_dna, device)
    scorer = load_scorer(config, config.model.alphabet_size, device)

    s = config.sampling
    seqs = sample_reglm(
        model, config.ctrl_dna,
        num_batches=s.num_batches, batch_size=s.batch_size,
        seq_length=s.seq_length, seed=seed,
    )
    mu, sigma = score_sequences(scorer, seqs)
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
