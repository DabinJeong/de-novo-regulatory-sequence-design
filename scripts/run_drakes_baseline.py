"""
DRAKES baseline runner.

Paper:
    Wang et al., "Reward-Directed Discrete Diffusion for DNA Sequence Design"
    (DRAKES), ICLR 2025.  https://github.com/ChenyuWang-Monica/DRAKES

What this script does
---------------------
1. Loads a DRAKES MDLM-based discrete diffusion checkpoint (finetuned /
   pretrained / zero_alpha / cfg) from an external DRAKES checkout.
2. Generates enhancer sequences via DRAKES' `Diffusion._sample()` API.
3. Writes a single-column CSV (`seq`) for downstream evaluation by
   `scripts.evaluate_sequences`, which scores with the independent
   DRAKES oracle (primary, non-circular) and the PropertyScorer
   (circular, diagnostic).

Prerequisites
-------------
- A checkout of https://github.com/ChenyuWang-Monica/DRAKES on disk, plus
  their Dropbox data+weights zip extracted to `drakes.base_path`.

CLI
---
    python -m scripts.run_drakes_baseline \
        --config configs/drakes_baseline.yaml \
        --out_dir ./runs/drakes_baseline
"""

import argparse
import csv
import os
import sys

import torch
import yaml
from ml_collections.config_dict import ConfigDict
from tqdm import tqdm


INT_TO_BASE = {0: "A", 1: "C", 2: "G", 3: "T"}


# ---------------------------------------------------------------------------
# DRAKES loader
# ---------------------------------------------------------------------------
def _resolve_ckpt_path(drakes_cfg) -> str:
    explicit = drakes_cfg.get("checkpoint_path", None)
    if explicit:
        return explicit
    base = drakes_cfg.base_path
    kind = drakes_cfg.get("checkpoint_kind", "finetuned")
    table = {
        "finetuned":  os.path.join(base, "mdlm/reward_bp_results_final/finetuned.ckpt"),
        "pretrained": os.path.join(base, "mdlm/outputs_gosai/pretrained.ckpt"),
        "zero_alpha": os.path.join(base, "mdlm/reward_bp_results_final/zero_alpha.ckpt"),
        "cfg":        os.path.join(base, "mdlm/outputs_gosai/cfg.ckpt"),
    }
    if kind not in table:
        raise ValueError(f"Unknown drakes.checkpoint_kind={kind!r}; "
                         f"use one of {list(table)} or set checkpoint_path.")
    return table[kind]


def load_drakes_model(drakes_cfg, device: torch.device):
    """Import DRAKES from disk and instantiate the requested Diffusion model."""
    repo_path = drakes_cfg.repo_path
    dna_dir = os.path.join(repo_path, "drakes_dna")
    if not os.path.isdir(dna_dir):
        raise FileNotFoundError(
            f"DRAKES drakes_dna/ not found at {dna_dir}. "
            "Clone https://github.com/ChenyuWang-Monica/DRAKES and set "
            "drakes.repo_path in the config."
        )
    # DRAKES uses relative imports + hydra with a local config dir, so we cd
    # into drakes_dna/ and add it to sys.path for the duration of model load.
    sys.path.insert(0, dna_dir)
    cwd_backup = os.getcwd()
    os.chdir(dna_dir)
    try:
        from hydra import compose, initialize_config_dir  # noqa: WPS433 (lazy import)
        from hydra.core.global_hydra import GlobalHydra
        import diffusion_gosai_update as diffusion_mod
        import diffusion_gosai_cfg as diffusion_cfg_mod

        # initialize() resolves config_path relative to the *calling file*, not
        # cwd, so we use initialize_config_dir with an absolute path inside the
        # DRAKES checkout.
        cfg_dir = os.path.abspath(
            os.path.join(dna_dir,
                         drakes_cfg.get("hydra_config_dir", "configs_gosai"))
        )
        GlobalHydra.instance().clear()
        initialize_config_dir(
            config_dir=cfg_dir,
            job_name="drakes_baseline",
            version_base=None,
        )
        cfg = compose(config_name=drakes_cfg.get("hydra_config_name",
                                                 "config_gosai.yaml"))

        ckpt_path = _resolve_ckpt_path(drakes_cfg)
        cfg.eval.checkpoint_path = ckpt_path
        kind = drakes_cfg.get("checkpoint_kind", "finetuned")

        if kind == "cfg":
            cfg.model.cls_free_guidance = True
            cfg.model.cls_free_weight = 10
            cfg.model.cls_free_prob = 0.1
            model = diffusion_cfg_mod.Diffusion(cfg, eval=False).to(device)
            sd = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(sd.get("state_dict", sd))
        elif kind == "pretrained":
            # DRAKES' pretrained ckpt uses Lightning's load_from_checkpoint path.
            model = diffusion_mod.Diffusion.load_from_checkpoint(
                ckpt_path, config=cfg
            ).to(device)
        else:  # finetuned / zero_alpha
            model = diffusion_mod.Diffusion(cfg, eval=False).to(device)
            sd = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(sd.get("state_dict", sd))

        model.eval()
        print(f"[DRAKES] loaded {kind} checkpoint from {ckpt_path}")
        return model, kind
    finally:
        os.chdir(cwd_backup)


# ---------------------------------------------------------------------------
# Sampling + scoring
# ---------------------------------------------------------------------------
@torch.no_grad()
def sample_drakes(model, kind: str, num_batches: int, batch_size: int) -> torch.Tensor:
    """Returns a (N, L) LongTensor of token ids."""
    samples = []
    for _ in tqdm(range(num_batches), desc=f"DRAKES[{kind}] sampling"):
        if kind == "cfg":
            out = model._sample(eval_sp_size=batch_size, w=10)
        else:
            out = model._sample(eval_sp_size=batch_size)
        samples.append(out.detach().cpu())
    return torch.cat(samples, dim=0)


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

    model, kind = load_drakes_model(config.drakes, device)

    s = config.sampling
    seqs = sample_drakes(model, kind, s.num_batches, s.batch_size)

    out_path = os.path.join(args.out_dir, "drakes_sequences.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seq"])
        for string in seqs_to_strings(seqs):
            writer.writerow([string])
    print(f"[drakes_baseline] wrote {seqs.size(0)} sequences -> {out_path}")


if __name__ == "__main__":
    main()
