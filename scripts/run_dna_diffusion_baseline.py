"""
DNA-Diffusion baseline runner.

Paper:
    Lal et al., "Designing synthetic regulatory elements using DNA-Diffusion",
    Nature Genetics 2025. Repo: https://github.com/pinellolab/DNA-Diffusion

What this script does
---------------------
1. Imports DNA-Diffusion from a local clone (using their Hydra configs).
2. Loads a pretrained cell-type-conditional diffusion checkpoint.
3. Calls the repo's own `model.sample(classes, shape, cond_weight)` routine
   per requested cell type, collecting (batch, 4, L) one-hot samples.
4. Converts the one-hots to token IDs (A=0, C=1, G=2, T=3).
5. Scores every sequence with our PropertyScorer (ensemble predictor).
6. Writes a CSV with columns `seq, mu, sigma, score, cell_type` matching
   the schema emitted by scripts/guided_sampler.py (plus a cell_type column
   so the downstream evaluator can filter per target).

CLI
---
    python -m scripts.run_dna_diffusion_baseline \
        --config configs/dna_diffusion_baseline.yaml \
        --out_dir ./runs/dna_diffusion_baseline
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch
import yaml
from ml_collections.config_dict import ConfigDict
from tqdm import tqdm

from sequence_generation.model.property_scorer import PropertyScorer


INT_TO_BASE = {0: "A", 1: "C", 2: "G", 3: "T"}
# DNA-Diffusion uses this nucleotide order; see sample_util.py.
NUCLEOTIDES = ["A", "C", "G", "T"]


# ---------------------------------------------------------------------------
# DNA-Diffusion loader
# ---------------------------------------------------------------------------
def load_dna_diffusion(dd_cfg, device: torch.device):
    """Import DNA-Diffusion from disk and build the sampler with hydra."""
    repo_path = dd_cfg.repo_path
    if not os.path.isdir(repo_path):
        raise FileNotFoundError(
            f"DNA-Diffusion repo not found at {repo_path}. "
            "Clone https://github.com/pinellolab/DNA-Diffusion and set "
            "dna_diffusion.repo_path."
        )
    src_path = os.path.join(repo_path, "src")
    sys.path.insert(0, src_path)
    cwd_backup = os.getcwd()
    os.chdir(repo_path)
    try:
        import hydra
        from hydra import compose, initialize
        from hydra.core.global_hydra import GlobalHydra
        from safetensors.torch import load_file

        GlobalHydra.instance().clear()
        initialize(
            config_path=dd_cfg.get("hydra_config_dir", "configs"),
            job_name="dna_diffusion_baseline",
            version_base=None,
        )
        cfg = compose(config_name=dd_cfg.get("hydra_config_name", "sample.yaml"))

        model = hydra.utils.instantiate(cfg.model)
        data = hydra.utils.instantiate(cfg.data)
        diffusion = hydra.utils.instantiate(cfg.diffusion, model=model)

        ckpt = dd_cfg.checkpoint_path
        if ckpt.endswith(".safetensors"):
            sd = load_file(ckpt) if device.type == "cuda" else load_file(ckpt, device="cpu")
            diffusion.model.load_state_dict(sd)
        else:
            sd = torch.load(ckpt, map_location=device)
            diffusion.model.load_state_dict(sd["model"] if "model" in sd else sd)

        diffusion = diffusion.to(device)
        diffusion.eval()
        print(f"[DNA-Diffusion] loaded {ckpt}")
        return diffusion, data
    finally:
        os.chdir(cwd_backup)


# ---------------------------------------------------------------------------
# Sampling + scoring
# ---------------------------------------------------------------------------
@torch.no_grad()
def sample_dna_diffusion(diffusion, data, cell_types, num_samples: int,
                         batch_size: int, guidance_scale: float,
                         seq_length: int = 200):
    """
    Mirrors dnadiffusion.utils.sample_util.create_sample but returns token
    IDs directly instead of writing a text file.
    Returns (all_ids, all_cts) — (N, L) LongTensor and (N,) list.
    """
    numeric_to_tag = data[-1]
    cell_num_list = data[-2] if cell_types is None else list(cell_types)

    device = next(diffusion.parameters()).device
    all_ids, all_cts = [], []
    num_batches = num_samples // batch_size
    for ct in cell_num_list:
        print(f"[DNA-Diffusion] sampling cell type {numeric_to_tag[ct]}")
        for _ in tqdm(range(num_batches), desc=f"ct={numeric_to_tag[ct]}"):
            classes = torch.full((batch_size,), ct, dtype=torch.long,
                                 device=device).float()
            sampled = diffusion.sample(
                classes, (batch_size, 1, 4, seq_length), guidance_scale,
            )
            final = sampled[-1] if isinstance(sampled, (list, tuple)) else sampled
            final = final.detach().cpu().numpy()
            # (B, 1, 4, L) or (B, 4, L) -> argmax along nucleotide axis
            if final.ndim == 4:
                final = final[:, 0]
            ids = final.argmax(axis=-2)                      # (B, L)
            all_ids.append(torch.from_numpy(ids).long())
            all_cts.extend([numeric_to_tag[ct]] * ids.shape[0])
    return torch.cat(all_ids, dim=0), all_cts


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
    device = next(scorer.parameters()).device
    for i in range(0, seqs.size(0), batch_size):
        chunk = seqs[i:i + batch_size].to(device)
        mu, var = scorer.mu_sigma2({"seqs": chunk})
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
    np.random.seed(config.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    diffusion, data = load_dna_diffusion(config.dna_diffusion, device)
    scorer = load_scorer(config, config.model.alphabet_size, device)

    s = config.sampling
    seqs, cts = sample_dna_diffusion(
        diffusion, data,
        cell_types=config.dna_diffusion.get("cell_types", None),
        num_samples=s.number_of_samples,
        batch_size=s.sample_batch_size,
        guidance_scale=config.dna_diffusion.guidance_scale,
    )
    mu, sigma = score_sequences(scorer, seqs)
    gamma = float(s.get("gamma_rank", 1.0))
    score = mu - gamma * sigma

    out_path = os.path.join(args.out_dir, "dna_diffusion_sequences.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seq", "mu", "sigma", "score", "cell_type"])
        for string, m, sg, sc, ct in zip(seqs_to_strings(seqs),
                                         mu.tolist(), sigma.tolist(),
                                         score.tolist(), cts):
            writer.writerow([string, f"{m:.4f}", f"{sg:.4f}", f"{sc:.4f}", ct])
    print(f"[dna_diffusion_baseline] wrote {seqs.size(0)} sequences -> {out_path}")


if __name__ == "__main__":
    main()
