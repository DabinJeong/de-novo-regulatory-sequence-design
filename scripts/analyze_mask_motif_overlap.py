"""
Motif-vs-mask overlap analysis for the masked separator.

Question: do JASPAR motif hits preferentially occupy env (mask-on) or
invariant (mask-off) positions? And is the env localization environment-specific?

For each sequence we:
    1. Scan the *full* sequence against JASPAR vertebrate-CORE PWMs (both
       strands, per-motif score threshold at a fixed FPR) using the same
       pipeline as ``evaluate_sequences.py``.
    2. For each hit spanning [start, start+W), compute the fraction of
       positions inside the hard mask (M >= threshold).
    3. Classify the hit as "env" (majority-mask overlap, >= 0.5) or
       "invariant" (< 0.5), and for env hits bucket by the sequence's
       env id inferred by K-means on embed_en(x_en) — same clustering as
       ``analyze_invariant_separator.py``.
    4. Aggregate hits/seq and hits/kb per region, and per (motif, region).

Why hits/kb matters: the invariant region is ~70 bp/seq while the env
region is ~130 bp/seq (from mask activation ~35 %), so raw hits/seq
favour env purely by length. Density normalises that away.

Why full-seq scan rather than substring scan (as in analyze_invariant_
separator.py): the mask is fragmented (median run ~ 1 bp, ~36 blocks/seq
at the current ckpt), so concatenating selected positions destroys motif
footprints. We scan the full sequence and score overlap instead.

Outputs
-------
    summary_region.csv       per region (invariant, env0..envK-1):
                              n_seqs, bp_total, total_hits, hits_per_seq,
                              hits_per_kb, coverage
    per_motif_region.csv     per (motif, region): hit count, hits_per_kb
    top_motifs_per_region.csv top-10 motifs by hits_per_kb for each region
    hit_overlap_hist.png     histogram of overlap fractions across all hits
    run_meta.json            parameters + aggregate numbers

Usage
-----
    python -m scripts.analyze_mask_motif_overlap \
        --config configs/enhancer_gosai_masked_separator.yaml \
        --ckpt   runs/masked_separator/masked_separator_best.ckpt \
        --out_dir runs/masked_separator/analysis/motif_overlap \
        --num_samples 1000 --split val
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from Bio.Seq import Seq
from ml_collections.config_dict import ConfigDict
from tqdm import tqdm

from scripts.analyze_invariant_separator import kmeans_fit_predict
from scripts.evaluate_sequences import _load_jaspar_pssms, _pssm_thresholds
from sequence_generation.model.masked_separator import MaskedSeparatorModel
from sequence_generation.utils.train_utils import load_dataloader


ID_TO_NUC = np.array(["A", "C", "G", "T"])


# ------------------------------------------------------------------
# Forward pass: collect tokens, masks, env ids
# ------------------------------------------------------------------
@torch.no_grad()
def collect_sequences_and_masks(
    model: MaskedSeparatorModel,
    loader,
    device: torch.device,
    num_samples: int,
    num_envs: int,
    centroids: torch.Tensor | None,
    kmeans_seed: int = 0,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    model.eval()
    tok_batches, M_batches, h_batches = [], [], []
    collected = 0
    for batch in tqdm(loader, desc="[forward] mask+tokens"):
        x_ids = batch["seqs"].to(device)
        M, _, x_en_soft = model.separate(x_ids, soft_input=False)
        h_en = model.embed_en(x_en_soft)
        tok_batches.append(x_ids.cpu())
        M_batches.append(M.cpu())
        h_batches.append(h_en.cpu())
        collected += x_ids.size(0)
        if collected >= num_samples:
            break

    tokens = torch.cat(tok_batches, dim=0)[:num_samples]
    M      = torch.cat(M_batches, dim=0)[:num_samples]
    H_en   = torch.cat(h_batches, dim=0)[:num_samples]

    if centroids is None:
        env_ids, _ = kmeans_fit_predict(H_en, K=num_envs, seed=kmeans_seed)
    else:
        d2 = torch.cdist(H_en, centroids.cpu(), p=2) ** 2
        env_ids = d2.argmin(dim=-1)

    seqs = ["".join(ID_TO_NUC[row]) for row in tokens.numpy()]
    return seqs, M.numpy(), env_ids.numpy()


# ------------------------------------------------------------------
# Motif scan: collect per-hit positions (not just counts)
# ------------------------------------------------------------------
def _hit_start(position: int, seq_len: int, W: int) -> int:
    """Forward-strand start coord for a Biopython PSSM.search hit.

    Biopython reports reverse-strand matches with a negative position;
    the forward-strand footprint of the motif occupies
    seq[abs(pos) : abs(pos) + W]. For forward-strand hits the position
    is already the start. We clamp to [0, seq_len - W] defensively.
    """
    start = position if position >= 0 else -position - 1
    return max(0, min(start, seq_len - W))


def scan_hits_with_positions(
    seqs: List[str],
    pssms: List[Tuple[str, object]],
    thresholds: List[float],
) -> List[List[Tuple[int, int, int]]]:
    """Return, per sequence, a list of hits (motif_idx, start, end)."""
    per_seq_hits: List[List[Tuple[int, int, int]]] = [[] for _ in seqs]
    for j, (_, pssm) in enumerate(tqdm(pssms, desc="[scan] motifs")):
        W = pssm.length
        thr = thresholds[j]
        for i, s in enumerate(seqs):
            if len(s) < W:
                continue
            bseq = Seq(s)
            for pos, _score in pssm.search(bseq, threshold=thr, both=True):
                start = _hit_start(pos, len(s), W)
                per_seq_hits[i].append((j, start, start + W))
    return per_seq_hits


# ------------------------------------------------------------------
# Classify hits by mask overlap, aggregate per region
# ------------------------------------------------------------------
def classify_and_aggregate(
    seqs: List[str],
    env_mask: np.ndarray,
    env_ids: np.ndarray,
    per_seq_hits: List[List[Tuple[int, int, int]]],
    motif_ids: List[str],
    num_envs: int,
    env_overlap_threshold: float = 0.5,
) -> Dict:
    """
    MaskedSeparator convention: M >= threshold selects the invariant (x_st)
    positions; M < threshold selects the env (x_en) positions. Here
    ``env_mask[i, l] == 1`` iff position l of sequence i is env-variable
    (i.e. M_soft < mask_threshold). For each motif hit spanning [s, e),

        overlap_frac = mean(env_mask[i, s:e])

    is the fraction of the footprint that falls in the env region. A hit is
    classified as env (and bucketed by the sequence's env_id) when
    overlap_frac >= env_overlap_threshold; otherwise invariant.
    """
    N = len(seqs)
    L = env_mask.shape[1]
    M_motifs = len(motif_ids)

    region_labels = ["invariant"] + [f"env{k}" for k in range(num_envs)]
    # region -> seqs considered for that region (for coverage + bp denom).
    # Invariant bp is shared across all sequences; env bp is specific to the
    # env-group each sequence was assigned to.
    region_seqs: Dict[str, List[int]] = {"invariant": list(range(N))}
    for k in range(num_envs):
        region_seqs[f"env{k}"] = list(np.where(env_ids == k)[0])

    bp_total: Dict[str, int] = {}
    bp_total["invariant"] = int((L - env_mask.sum(axis=1)).sum())
    for k in range(num_envs):
        sel = env_ids == k
        bp_total[f"env{k}"] = int(env_mask[sel].sum()) if sel.any() else 0

    # Per-seq hit totals per region, and per-motif-per-region counts.
    per_seq_region_hits = {lab: np.zeros(N, dtype=np.int64) for lab in region_labels}
    per_motif_region_hits = {
        lab: np.zeros(M_motifs, dtype=np.int64) for lab in region_labels
    }
    overlap_fracs: List[float] = []

    for i, hits in enumerate(per_seq_hits):
        env_k = int(env_ids[i])
        for (j, s, e) in hits:
            frac = float(env_mask[i, s:e].mean())
            overlap_fracs.append(frac)
            if frac >= env_overlap_threshold:
                lab = f"env{env_k}"
            else:
                lab = "invariant"
            per_seq_region_hits[lab][i] += 1
            per_motif_region_hits[lab][j] += 1

    # Per-region summary
    rows = []
    for lab in region_labels:
        seq_idx = region_seqs[lab]
        hits_vec = per_seq_region_hits[lab][seq_idx]
        bp = bp_total[lab]
        rows.append({
            "region": lab,
            "n_seqs": len(seq_idx),
            "bp_total": bp,
            "total_hits": int(hits_vec.sum()),
            "hits_per_seq_mean": float(hits_vec.mean()) if len(seq_idx) else np.nan,
            "hits_per_seq_median": float(np.median(hits_vec)) if len(seq_idx) else np.nan,
            "hits_per_kb": (1000.0 * hits_vec.sum() / bp) if bp > 0 else np.nan,
            "coverage": float((hits_vec > 0).mean()) if len(seq_idx) else np.nan,
        })
    summary_df = pd.DataFrame(rows)

    # Per-(motif, region) table: hits + hits/kb
    motif_rows = []
    for lab in region_labels:
        bp = bp_total[lab]
        for mi, mid in enumerate(motif_ids):
            h = int(per_motif_region_hits[lab][mi])
            motif_rows.append({
                "motif_id": mid,
                "region": lab,
                "hits": h,
                "hits_per_kb": (1000.0 * h / bp) if bp > 0 else np.nan,
            })
    per_motif_df = pd.DataFrame(motif_rows)

    return {
        "summary_df": summary_df,
        "per_motif_df": per_motif_df,
        "overlap_fracs": np.array(overlap_fracs, dtype=float),
        "bp_total": bp_total,
    }


def top_motifs_per_region(per_motif_df: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    out = []
    for lab, g in per_motif_df.groupby("region"):
        gg = g.sort_values("hits_per_kb", ascending=False).head(top_k).copy()
        gg["rank"] = np.arange(1, len(gg) + 1)
        out.append(gg)
    return pd.concat(out, axis=0).reset_index(drop=True)


# ------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------
def plot_overlap_hist(fracs: np.ndarray, out_path: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    if fracs.size:
        ax.hist(fracs, bins=np.linspace(0.0, 1.0, 21), edgecolor="black")
        ax.axvline(0.5, color="red", ls="--", lw=1,
                   label="env/invariant split (0.5)")
        ax.set_xlabel("fraction of motif footprint inside hard mask (env)")
        ax.set_ylabel("# motif hits")
        ax.set_title(f"Hit–mask overlap distribution (n={fracs.size})")
        ax.legend()
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
    ap.add_argument("--env_overlap_threshold", type=float, default=0.5,
                    help="Hit is counted as 'env' if >= this fraction of "
                         "its footprint is inside the hard mask.")
    ap.add_argument("--jaspar_release", type=str, default="JASPAR2022")
    ap.add_argument("--jaspar_collection", type=str, default="CORE")
    ap.add_argument("--jaspar_tax_group", type=str, default="vertebrates")
    ap.add_argument("--jaspar_fpr", type=float, default=1e-3)
    ap.add_argument("--jaspar_max_motifs", type=int, default=None,
                    help="Cap # PWMs (for smoke tests). Unset = all.")
    ap.add_argument("--top_k_motifs", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.config) as f:
        config = ConfigDict(yaml.safe_load(f))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = load_dataloader(config)
    loader = {"train": train_loader, "val": val_loader, "test": test_loader}[args.split]

    # Model + checkpoint
    model = MaskedSeparatorModel(config).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state)
    print(f"[ckpt] loaded {args.ckpt}")

    centroids = None
    if isinstance(ckpt, dict) and ckpt.get("env_centroids") is not None:
        centroids = ckpt["env_centroids"].to(device)
        print(f"[env] using stored centroids with shape {tuple(centroids.shape)}")

    de_cfg = config.get("masked_separator", config.get("dual_encoder", {}))
    num_envs = de_cfg.get("num_envs", 3)

    # 1) forward pass
    seqs, M_soft, env_ids = collect_sequences_and_masks(
        model, loader, device,
        num_samples=args.num_samples,
        num_envs=num_envs,
        centroids=centroids,
    )
    # MaskedSeparator convention: M >= threshold => invariant (x_st).
    # We care about env-region overlap, so env_mask=1 at env-variable
    # positions (M < threshold) and 0 at invariant positions.
    env_mask = (M_soft < args.mask_threshold).astype(np.int8)
    print(f"[mask] N={len(seqs)}  L={env_mask.shape[1]}  "
          f"mean(env)={env_mask.mean():.3f}  "
          f"env_sizes={[int((env_ids == k).sum()) for k in range(num_envs)]}")

    # 2) JASPAR PWMs + thresholds
    pssms = _load_jaspar_pssms(
        release=args.jaspar_release,
        collection=args.jaspar_collection,
        tax_group=args.jaspar_tax_group,
        pseudocount=0.5,
        max_motifs=args.jaspar_max_motifs,
    )
    motif_ids = [mid for mid, _ in pssms]
    print(f"[jaspar] loaded {len(pssms)} PWMs  (release={args.jaspar_release}, "
          f"collection={args.jaspar_collection}, tax={args.jaspar_tax_group})")
    background = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}
    thresholds = _pssm_thresholds(pssms, background, args.jaspar_fpr)

    # 3) per-sequence hits with positions
    per_seq_hits = scan_hits_with_positions(seqs, pssms, thresholds)
    n_hits = sum(len(h) for h in per_seq_hits)
    print(f"[scan] total hits across {len(seqs)} seqs: {n_hits}  "
          f"({n_hits / max(len(seqs), 1):.1f}/seq)")

    # 4) classify + aggregate
    agg = classify_and_aggregate(
        seqs, env_mask, env_ids, per_seq_hits, motif_ids, num_envs,
        env_overlap_threshold=args.env_overlap_threshold,
    )
    summary_df = agg["summary_df"]
    per_motif_df = agg["per_motif_df"]

    summary_df.to_csv(os.path.join(args.out_dir, "summary_region.csv"), index=False)
    per_motif_df.to_csv(os.path.join(args.out_dir, "per_motif_region.csv"), index=False)
    top_motifs_per_region(per_motif_df, top_k=args.top_k_motifs).to_csv(
        os.path.join(args.out_dir, "top_motifs_per_region.csv"), index=False)

    plot_overlap_hist(agg["overlap_fracs"],
                      os.path.join(args.out_dir, "hit_overlap_hist.png"))

    print("\n[summary per region]")
    print(summary_df.round(3).to_string(index=False))

    with open(os.path.join(args.out_dir, "run_meta.json"), "w") as f:
        json.dump({
            "ckpt": args.ckpt,
            "split": args.split,
            "num_samples": len(seqs),
            "seq_length": int(env_mask.shape[1]),
            "mask_threshold": args.mask_threshold,
            "env_overlap_threshold": args.env_overlap_threshold,
            "num_envs": num_envs,
            "env_sizes": [int((env_ids == k).sum()) for k in range(num_envs)],
            "mean_env_fraction": float(env_mask.mean()),
            "mean_invariant_fraction": float(1.0 - env_mask.mean()),
            "jaspar_release": args.jaspar_release,
            "jaspar_collection": args.jaspar_collection,
            "jaspar_tax_group": args.jaspar_tax_group,
            "jaspar_fpr": args.jaspar_fpr,
            "num_motifs": len(motif_ids),
            "total_hits": int(n_hits),
            "bp_total": agg["bp_total"],
        }, f, indent=2)

    print(f"\n[done] outputs written to {args.out_dir}")


if __name__ == "__main__":
    main()
