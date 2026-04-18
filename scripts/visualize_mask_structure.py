"""
Visualise the spatial structure of the masked separator's invariant mask.

Answers: in each sequence, are the positions selected into x_st (M >= threshold)
one contiguous island, or scattered into many short blocks?

Outputs (under --out_dir):
    mask_heatmap.png         soft-mask values for the first N_rows sequences
                             sorted by mean mask (row = sequence, col = position)
    per_position_stats.png   mean(M) and selection frequency across samples
    run_length_hist.png      histogram of contiguous run lengths of M >= threshold
    num_blocks_hist.png      # of disjoint invariant blocks per sequence
    autocorrelation.png      lag-wise Pearson r of the hard mask along position
    per_sequence_stats.csv   per-sequence summary numbers
    run_meta.json            parameters + aggregate numbers

Usage
-----
    python -m scripts.visualize_mask_structure \
        --config configs/enhancer_gosai_masked_separator.yaml \
        --ckpt   runs/masked_separator/masked_separator_best.ckpt \
        --out_dir runs/masked_separator/analysis/mask_structure \
        --num_samples 1000 --split val
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from ml_collections.config_dict import ConfigDict
from tqdm import tqdm

from sequence_generation.model.masked_separator import MaskedSeparatorModel
from sequence_generation.utils.train_utils import load_dataloader


# ------------------------------------------------------------------
@torch.no_grad()
def collect_masks(model, loader, device, num_samples):
    model.eval()
    chunks = []
    n = 0
    for batch in tqdm(loader, desc="[forward]"):
        x = batch["seqs"].to(device)
        M, _, _ = model.separate(x, soft_input=False)
        chunks.append(M.cpu())
        n += x.size(0)
        if n >= num_samples:
            break
    M = torch.cat(chunks, dim=0)[:num_samples]        # (N, L)
    return M.numpy()


# ------------------------------------------------------------------
# Run-length / block analysis on a binary mask
# ------------------------------------------------------------------
def run_lengths(binary_row: np.ndarray) -> List[int]:
    """Lengths of contiguous stretches of 1 in a 0/1 vector."""
    if binary_row.sum() == 0:
        return []
    diff = np.diff(np.concatenate([[0], binary_row.astype(int), [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return (ends - starts).tolist()


def per_sequence_block_stats(hard: np.ndarray) -> pd.DataFrame:
    """
    hard: (N, L) bool/0-1 array.
    Returns per-sequence: frac_selected, n_blocks, longest_run, median_run,
    gap_mean (mean gap between blocks).
    """
    rows = []
    L = hard.shape[1]
    for i in range(hard.shape[0]):
        row = hard[i].astype(int)
        rl = run_lengths(row)
        gaps = run_lengths(1 - row)
        # gaps includes the leading and trailing 0-runs; strip those if present
        # (we only care about between-block gaps).
        if len(rl) >= 2:
            between = gaps
            # remove boundary gaps (leading/trailing zeros)
            if row[0] == 0:
                between = between[1:]
            if row[-1] == 0 and between:
                between = between[:-1]
            gap_mean = float(np.mean(between)) if between else 0.0
        else:
            gap_mean = np.nan
        rows.append({
            "frac_selected": row.sum() / L,
            "n_blocks":       len(rl),
            "longest_run":    int(max(rl)) if rl else 0,
            "median_run":     float(np.median(rl)) if rl else 0.0,
            "gap_mean":       gap_mean,
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Autocorrelation of the hard mask along the position axis
# ------------------------------------------------------------------
def mask_autocorrelation(hard: np.ndarray, max_lag: int = 50) -> np.ndarray:
    """
    For each lag k in [1, max_lag], compute per-sample Pearson r between
    hard[:, :-k] and hard[:, k:], then average across samples.
    r(0) = 1.  r(k) staying high -> contiguous; dropping fast -> scattered.
    """
    N, L = hard.shape
    H = hard.astype(float)
    H = H - H.mean(axis=1, keepdims=True)
    std = H.std(axis=1, keepdims=True) + 1e-12
    H = H / std
    out = np.zeros(max_lag + 1)
    out[0] = 1.0
    for k in range(1, max_lag + 1):
        a = H[:, :-k]
        b = H[:, k:]
        r_per_sample = (a * b).mean(axis=1)
        out[k] = np.nanmean(r_per_sample)
    return out


# ------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------
def plot_mask_heatmap(M: np.ndarray, out_path: str, n_rows: int = 200,
                     threshold: float = 0.5):
    order = np.argsort(-M.mean(axis=1))
    show = M[order[:n_rows]]
    hard = (show >= threshold).astype(float)
    
    fig, ax = plt.subplots(figsize=(10, max(3, n_rows * 0.03)))
    im = ax.imshow(hard, aspect="auto", cmap="binary", vmin=0, vmax=1,
                   interpolation="nearest")
    ax.set_xlabel("position")
    ax.set_ylabel(f"sequence (top {n_rows} by mean M)")
    ax.set_title(f"Hard mask (M >= {threshold})")
    plt.colorbar(im, ax=ax, label="selected")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_position(M: np.ndarray, hard: np.ndarray, out_path: str):
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axes[0].plot(M.mean(axis=0))
    axes[0].set_ylabel("mean soft M")
    axes[0].set_title("Per-position mask statistics")
    axes[0].axhline(0.5, ls="--", lw=0.5, color="k")
    axes[1].plot(hard.mean(axis=0), color="C1")
    axes[1].set_ylabel("selection frequency")
    axes[1].set_xlabel("position")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_run_length_hist(rl_all: List[int], out_path: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    if rl_all:
        max_rl = max(rl_all)
        bins = np.arange(1, max_rl + 2)
        ax.hist(rl_all, bins=bins, edgecolor="black")
        ax.set_xlabel("contiguous run length (bp)")
        ax.set_ylabel("count")
        ax.set_title(f"Invariant-block run lengths (n={len(rl_all)} blocks)")
        ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_num_blocks_hist(n_blocks: np.ndarray, out_path: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(n_blocks, bins=np.arange(0, int(n_blocks.max()) + 2),
            edgecolor="black")
    ax.set_xlabel("# disjoint invariant blocks per sequence")
    ax.set_ylabel("# sequences")
    median = np.median(n_blocks)
    ax.axvline(median, ls="--", color="red", label=f"median = {median:.0f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_autocorr(acorr: np.ndarray, out_path: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(len(acorr)), acorr, marker="o", ms=3)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("lag (positions)")
    ax.set_ylabel("mean Pearson r of hard mask")
    ax.set_title("Spatial autocorrelation of the hard mask")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_samples", type=int, default=1000)
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--mask_threshold", type=float, default=0.5)
    ap.add_argument("--heatmap_rows", type=int, default=200)
    ap.add_argument("--max_lag", type=int, default=50)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.config) as f:
        config = ConfigDict(yaml.safe_load(f))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = load_dataloader(config)
    loader = {"train": train_loader, "val": val_loader, "test": test_loader}[args.split]

    model = MaskedSeparatorModel(config).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    print(f"[ckpt] loaded {args.ckpt}")

    # 1) forward pass
    M = collect_masks(model, loader, device, args.num_samples)
    hard = (M >= args.mask_threshold)
    N, L = M.shape
    print(f"[mask] N={N}  L={L}  mean(M)={M.mean():.3f}  "
          f"mean(hard)={hard.mean():.3f}")

    # 2) per-sequence block analysis
    per_seq = per_sequence_block_stats(hard)
    per_seq.to_csv(os.path.join(args.out_dir, "per_sequence_stats.csv"), index=False)
    print("\n[per-sequence summary]\n", per_seq.describe().round(3))

    # Collect all run lengths across sequences (for the histogram)
    rl_all: List[int] = []
    for i in range(hard.shape[0]):
        rl_all.extend(run_lengths(hard[i].astype(int)))

    # 3) autocorrelation
    acorr = mask_autocorrelation(hard, max_lag=args.max_lag)

    # 4) figures
    plot_mask_heatmap(M, os.path.join(args.out_dir, "mask_heatmap.png"),
                  n_rows=min(args.heatmap_rows, N),
                  threshold=args.mask_threshold)
    plot_per_position(M, hard, os.path.join(args.out_dir, "per_position_stats.png"))
    plot_run_length_hist(rl_all, os.path.join(args.out_dir, "run_length_hist.png"))
    plot_num_blocks_hist(per_seq["n_blocks"].values,
                         os.path.join(args.out_dir, "num_blocks_hist.png"))
    plot_autocorr(acorr, os.path.join(args.out_dir, "autocorrelation.png"))

    # 5) meta
    with open(os.path.join(args.out_dir, "run_meta.json"), "w") as f:
        json.dump({
            "ckpt": args.ckpt,
            "split": args.split,
            "num_samples": int(N),
            "seq_length": int(L),
            "mask_threshold": args.mask_threshold,
            "mean_soft_M": float(M.mean()),
            "mean_selection_freq": float(hard.mean()),
            "median_blocks_per_seq": float(per_seq["n_blocks"].median()),
            "mean_blocks_per_seq":   float(per_seq["n_blocks"].mean()),
            "median_longest_run":    float(per_seq["longest_run"].median()),
            "median_run_length":     float(per_seq["median_run"].median()),
            "autocorr_lag1":         float(acorr[1]),
            "autocorr_lag10":        float(acorr[min(10, len(acorr) - 1)]),
        }, f, indent=2)

    print(f"\n[done] figures + csv written to {args.out_dir}")


if __name__ == "__main__":
    main()
