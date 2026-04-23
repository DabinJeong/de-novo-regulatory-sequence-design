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
5. Writes a CSV with columns `seq, cell_type` for downstream evaluation by
   `scripts.evaluate_sequences`. The evaluator scores with the independent
   DRAKES oracle (primary, non-circular) and PropertyScorer (circular,
   diagnostic); the cell_type column lets the evaluator filter per target.

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
    # DNA-Diffusion's hydra configs reference targets as `src.dnadiffusion....`,
    # so the *repo root* (parent of src/) must be on sys.path for `src` to
    # resolve as a package. Also keep src/ itself for any `dnadiffusion.*`
    # imports used internally.
    sys.path.insert(0, repo_path)
    sys.path.insert(0, os.path.join(repo_path, "src"))
    cwd_backup = os.getcwd()
    os.chdir(repo_path)
    try:
        import hydra
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from safetensors.torch import load_file

        # initialize() resolves config_path relative to the *calling file*, not
        # cwd, so we use initialize_config_dir with an absolute path inside the
        # DNA-Diffusion checkout.
        cfg_dir = os.path.abspath(
            os.path.join(repo_path, dd_cfg.get("hydra_config_dir", "configs"))
        )
        GlobalHydra.instance().clear()
        initialize_config_dir(
            config_dir=cfg_dir,
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
        else:
            sd = torch.load(ckpt, map_location=device)
            sd = sd.get("state_dict", sd.get("model", sd))

        # DNA-Diffusion checkpoints are saved from the Diffusion wrapper, which
        # holds the UNet as `self.model`. Strip the "model." prefix so keys
        # align with the inner UNet we're loading into.
        if sd and all(k.startswith("model.") for k in sd.keys()):
            sd = {k[len("model."):]: v for k, v in sd.items()}
        diffusion.model.load_state_dict(sd)

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
            # DNA-Diffusion's sample() can return a tensor or a numpy array
            # depending on the code path; handle both.
            if torch.is_tensor(final):
                final = final.detach().cpu().numpy()
            else:
                final = np.asarray(final)
            # (B, 1, 4, L) or (B, 4, L) -> argmax along nucleotide axis
            if final.ndim == 4:
                final = final[:, 0]
            ids = final.argmax(axis=-2)                      # (B, L)
            all_ids.append(torch.from_numpy(ids).long())
            all_cts.extend([numeric_to_tag[ct]] * ids.shape[0])
    return torch.cat(all_ids, dim=0), all_cts


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

    s = config.sampling
    seqs, cts = sample_dna_diffusion(
        diffusion, data,
        cell_types=config.dna_diffusion.get("cell_types", None),
        num_samples=s.number_of_samples,
        batch_size=s.sample_batch_size,
        guidance_scale=config.dna_diffusion.guidance_scale,
    )

    out_path = os.path.join(args.out_dir, "dna_diffusion_sequences.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seq", "cell_type"])
        for string, ct in zip(seqs_to_strings(seqs), cts):
            writer.writerow([string, ct])
    print(f"[dna_diffusion_baseline] wrote {seqs.size(0)} sequences -> {out_path}")


if __name__ == "__main__":
    main()
