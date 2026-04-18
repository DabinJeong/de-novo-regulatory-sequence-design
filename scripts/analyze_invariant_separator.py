"""
Empirical validation of the masked separator: is x_st (invariant) actually
different from the per-environment x_en (background)?

4 groups compared:
    - invariant : x_st = positions where mask M >= threshold (pooled over all samples)
    - env_k     : x_en = positions where M < threshold, grouped by the K-means
                  env id inferred from embed_en(x_en)

For each group we extract one variable-length nucleotide substring per sample
and compute:
    * composition      : GC%, per-base freq
    * k-mer            : 3-mer, 6-mer distributions (PCA + Jensen-Shannon)
    * motif            : optional PWM scan from a JASPAR MEME file
    * complexity       : Shannon entropy, low-complexity window fraction
    * length           : length of the sub-string per sample

Statistics (per scalar metric):
    * 4-group test     : Kruskal-Wallis
    * pairwise         : Mann-Whitney U + Cliff's delta
    * k-mer pooled     : chi-square across groups
    * multiple testing : Benjamini-Hochberg on pairwise p-values

Usage
-----
    python -m scripts.analyze_invariant_separator \
        --config configs/enhancer_gosai_masked_separator.yaml \
        --ckpt   runs/masked_separator/masked_separator_best.ckpt \
        --out_dir runs/masked_separator/analysis \
        --num_samples 4000 \
        [--meme_file path/to/jaspar.meme]      # optional
        [--split val]                          # train | val | test
        [--mask_threshold 0.5]
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from ml_collections.config_dict import ConfigDict
from scipy.stats import (
    chi2_contingency,
    kruskal,
    mannwhitneyu,
)
from sklearn.decomposition import PCA
from tqdm import tqdm

from sequence_generation.model.masked_separator import MaskedSeparatorModel
from sequence_generation.utils.train_utils import load_dataloader


# ------------------------------------------------------------------
# Alphabet
# ------------------------------------------------------------------
ID_TO_NUC = np.array(["A", "C", "G", "T"])
ALPHA = ["A", "C", "G", "T"]


# ------------------------------------------------------------------
# K-means (matches trainer's implementation)
# ------------------------------------------------------------------
@torch.no_grad()
def kmeans_fit_predict(
    X: torch.Tensor, K: int, n_iters: int = 20, seed: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    N, _ = X.shape
    g = torch.Generator(device="cpu").manual_seed(seed)
    init_idx = torch.randperm(N, generator=g)[:K].to(X.device)
    centroids = X[init_idx].clone()
    for _ in range(n_iters):
        d2 = torch.cdist(X, centroids, p=2) ** 2
        assign = d2.argmin(dim=-1)
        new_c = centroids.clone()
        for k in range(K):
            m = assign == k
            if m.sum() > 0:
                new_c[k] = X[m].mean(dim=0)
        if (new_c - centroids).norm() < 1e-5:
            centroids = new_c
            break
        centroids = new_c
    return assign, centroids


# ------------------------------------------------------------------
# Group extraction
# ------------------------------------------------------------------
@torch.no_grad()
def extract_groups(
    model: MaskedSeparatorModel,
    loader,
    device: torch.device,
    num_samples: int,
    threshold: float,
    num_envs: int,
    centroids: Optional[torch.Tensor] = None,
    kmeans_seed: int = 0,
) -> Dict:
    """
    Returns dict with:
        invariant_seqs : list[str]         (len N_collected)
        env_seqs       : list[list[str]]   (num_envs groups of strings)
        M_mean_per_sample : np.ndarray     (N,)  per-sample mean mask value
        env_ids        : np.ndarray        (N,)
    """
    model.eval()
    token_batches, M_batches, en_embed_batches = [], [], []
    collected = 0

    for batch in tqdm(loader, desc="[forward] extract"):
        x_ids = batch["seqs"].to(device)                         # (B, L)
        M, x_st_soft, x_en_soft = model.separate(x_ids, soft_input=False)
        h_en = model.embed_en(x_en_soft)                         # (B, H)
        token_batches.append(x_ids.cpu())
        M_batches.append(M.cpu())
        en_embed_batches.append(h_en.cpu())
        collected += x_ids.size(0)
        if collected >= num_samples:
            break

    tokens = torch.cat(token_batches, dim=0)[:num_samples]       # (N, L)
    M      = torch.cat(M_batches, dim=0)[:num_samples]           # (N, L)
    H_en   = torch.cat(en_embed_batches, dim=0)[:num_samples]    # (N, Hdim)

    # K-means env ids
    if centroids is None:
        env_ids, centroids = kmeans_fit_predict(H_en, K=num_envs, seed=kmeans_seed)
    else:
        d2 = torch.cdist(H_en, centroids.cpu(), p=2) ** 2
        env_ids = d2.argmin(dim=-1)
    env_ids = env_ids.numpy()

    # Per-sample substrings
    hard_mask = (M >= threshold).numpy()                         # (N, L)
    tok_np = tokens.numpy()

    invariant_seqs: List[str] = []
    env_seqs: List[List[str]] = [[] for _ in range(num_envs)]
    for i in range(tok_np.shape[0]):
        st_pos = hard_mask[i]
        en_pos = ~st_pos
        s_st = "".join(ID_TO_NUC[tok_np[i, st_pos]]) if st_pos.any() else ""
        s_en = "".join(ID_TO_NUC[tok_np[i, en_pos]]) if en_pos.any() else ""
        if s_st:
            invariant_seqs.append(s_st)
        if s_en:
            env_seqs[int(env_ids[i])].append(s_en)

    return {
        "invariant_seqs": invariant_seqs,
        "env_seqs": env_seqs,
        "M_mean": M.mean(dim=-1).numpy(),
        "env_ids": env_ids,
        "hard_mask_mean": hard_mask.mean(),
    }


# ------------------------------------------------------------------
# Per-sequence statistics
# ------------------------------------------------------------------
def base_counts(s: str) -> np.ndarray:
    c = Counter(s)
    return np.array([c.get(b, 0) for b in ALPHA], dtype=float)


def gc_content(s: str) -> float:
    if not s:
        return np.nan
    return (s.count("G") + s.count("C")) / len(s)


def shannon_entropy(s: str) -> float:
    if not s:
        return np.nan
    c = base_counts(s)
    p = c / c.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def low_complexity_fraction(s: str, window: int = 20, ent_thresh: float = 1.5) -> float:
    """Fraction of the sequence inside a sliding window of low nucleotide entropy."""
    L = len(s)
    if L < window:
        return np.nan
    flags = np.zeros(L, dtype=bool)
    for i in range(L - window + 1):
        sub = s[i : i + window]
        c = base_counts(sub)
        p = c / c.sum()
        p = p[p > 0]
        ent = -(p * np.log2(p)).sum()
        if ent < ent_thresh:
            flags[i : i + window] = True
    return float(flags.mean())


def kmer_counts(s: str, k: int) -> np.ndarray:
    """Counts of all 4**k k-mers (alphabet ACGT) in s, in fixed lexicographic order."""
    dim = 4 ** k
    v = np.zeros(dim, dtype=np.int64)
    if len(s) < k:
        return v
    idx_map = {"A": 0, "C": 1, "G": 2, "T": 3}
    cur = 0
    valid = 0
    mask = dim - 1
    for ch in s:
        if ch not in idx_map:
            valid = 0
            cur = 0
            continue
        cur = ((cur << 2) | idx_map[ch]) & mask
        valid += 1
        if valid >= k:
            v[cur] += 1
    return v


def per_sample_stats(seqs: List[str], k_small: int = 3) -> pd.DataFrame:
    rows = []
    for s in seqs:
        bc = base_counts(s)
        tot = bc.sum() if bc.sum() > 0 else np.nan
        rows.append({
            "length":   len(s),
            "gc":       gc_content(s),
            "A":        bc[0] / tot if tot else np.nan,
            "C":        bc[1] / tot if tot else np.nan,
            "G":        bc[2] / tot if tot else np.nan,
            "T":        bc[3] / tot if tot else np.nan,
            "entropy":  shannon_entropy(s),
            "low_complexity": low_complexity_fraction(s),
        })
    df = pd.DataFrame(rows)
    # per-sample 3-mer distribution (for PCA)
    kmat = np.stack([kmer_counts(s, k_small) for s in seqs])      # (N, 64)
    kmat = kmat / np.clip(kmat.sum(axis=1, keepdims=True), 1, None)
    df_km = pd.DataFrame(kmat, columns=[f"kmer3_{i}" for i in range(kmat.shape[1])])
    return pd.concat([df, df_km], axis=1)


# ------------------------------------------------------------------
# Divergences
# ------------------------------------------------------------------
def jensen_shannon(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    def kl(a, b):
        return float(np.sum(a * (np.log2(a) - np.log2(b))))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def group_kmer_distribution(seqs: List[str], k: int) -> np.ndarray:
    dim = 4 ** k
    total = np.zeros(dim, dtype=np.int64)
    for s in seqs:
        total += kmer_counts(s, k)
    return total.astype(float)


# ------------------------------------------------------------------
# Effect size: Cliff's delta
# ------------------------------------------------------------------
def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cliff's delta = P(X>Y) - P(X<Y), X~a, Y~b.
    Implemented via rank of the combined sample (O((n+m) log (n+m))).
    """
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    n, m = len(a), len(b)
    combined = np.concatenate([a, b])
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, n + m + 1)
    # average ranks for ties
    # simple fallback: use scipy.stats.rankdata? keep dependency-light
    from scipy.stats import rankdata
    ranks = rankdata(combined, method="average")
    ra = ranks[:n]
    # U = sum(ra) - n(n+1)/2 ; P(X>Y) = U/(nm); delta = 2P - 1 (adjusting ties)
    U_a = ra.sum() - n * (n + 1) / 2.0
    return float(2.0 * U_a / (n * m) - 1.0)


# ------------------------------------------------------------------
# Benjamini-Hochberg
# ------------------------------------------------------------------
def bh_correct(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1)
    # enforce monotonicity
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0, 1)
    return out


# ------------------------------------------------------------------
# Optional: PWM / motif scan from a MEME file
# ------------------------------------------------------------------
def parse_meme_pwms(meme_path: str) -> List[Tuple[str, np.ndarray]]:
    """Minimal MEME parser. Returns [(motif_name, PWM (W,4)), ...] in ACGT order."""
    motifs = []
    with open(meme_path) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("MOTIF"):
            parts = line.split()
            name = parts[1] if len(parts) > 1 else f"motif_{len(motifs)}"
            # find "letter-probability matrix" line
            j = i + 1
            while j < len(lines) and not lines[j].startswith("letter-probability"):
                j += 1
            if j == len(lines):
                break
            header = lines[j]
            w = 0
            for tok in header.split():
                if tok.startswith("w="):
                    w = int(tok.split("=")[1])
            rows = []
            k = j + 1
            while k < len(lines) and len(rows) < w:
                vals = lines[k].split()
                if len(vals) >= 4:
                    rows.append([float(v) for v in vals[:4]])
                k += 1
            if len(rows) == w:
                motifs.append((name, np.array(rows)))
            i = k
        else:
            i += 1
    return motifs


def scan_motif_hits(seqs: List[str], pwms, threshold: float = 0.8) -> np.ndarray:
    """Count motif occurrences per sequence (max-hit score normalized by max-pwm)."""
    N = len(seqs)
    M = len(pwms)
    hits = np.zeros((N, M), dtype=np.int64)
    idx_map = {"A": 0, "C": 1, "G": 2, "T": 3}
    for mi, (_, pwm) in enumerate(pwms):
        W = pwm.shape[0]
        log_pwm = np.log(np.clip(pwm, 1e-6, 1.0))
        max_score = log_pwm.max(axis=1).sum()
        min_score = log_pwm.min(axis=1).sum()
        span = max_score - min_score + 1e-9
        for si, s in enumerate(seqs):
            if len(s) < W:
                continue
            arr = np.array([idx_map.get(c, -1) for c in s], dtype=np.int64)
            cnt = 0
            for p in range(len(s) - W + 1):
                sub = arr[p : p + W]
                if (sub < 0).any():
                    continue
                score = log_pwm[np.arange(W), sub].sum()
                if (score - min_score) / span >= threshold:
                    cnt += 1
            hits[si, mi] = cnt
    return hits


# ------------------------------------------------------------------
# Statistical testing driver
# ------------------------------------------------------------------
def run_tests(
    group_labels: List[str],
    group_dfs: List[pd.DataFrame],
    scalar_cols: List[str],
) -> pd.DataFrame:
    """
    Kruskal-Wallis across all groups + pairwise Mann-Whitney U + Cliff's delta.
    BH correction is applied to pairwise p-values per metric.
    """
    rows = []

    # 4-group Kruskal
    for col in scalar_cols:
        arrays = [df[col].dropna().values for df in group_dfs]
        if all(len(a) >= 5 for a in arrays):
            stat, p = kruskal(*arrays)
        else:
            stat, p = np.nan, np.nan
        rows.append({
            "metric": col, "test": "kruskal", "groups": "|".join(group_labels),
            "stat": stat, "p": p, "p_adj": np.nan, "effect": np.nan,
        })

    # Pairwise Mann-Whitney + Cliff's delta (per-metric BH)
    pair_records = []
    for col in scalar_cols:
        pvals, keys = [], []
        for i, j in itertools.combinations(range(len(group_dfs)), 2):
            a = group_dfs[i][col].dropna().values
            b = group_dfs[j][col].dropna().values
            if len(a) < 5 or len(b) < 5:
                p, stat = np.nan, np.nan
                eff = np.nan
            else:
                try:
                    stat, p = mannwhitneyu(a, b, alternative="two-sided")
                except ValueError:
                    stat, p = np.nan, 1.0
                eff = cliffs_delta(a, b)
            keys.append((col, group_labels[i], group_labels[j], stat, p, eff))
            pvals.append(p)
        p_adj = bh_correct(np.array(pvals, dtype=float))
        for (col_, a_, b_, st_, p_, eff_), pa_ in zip(keys, p_adj):
            pair_records.append({
                "metric": col_, "test": "mwu", "groups": f"{a_}|{b_}",
                "stat": st_, "p": p_, "p_adj": pa_, "effect": eff_,
            })
    return pd.DataFrame(rows + pair_records)


def kmer_chi2_across_groups(group_seqs: List[List[str]], labels: List[str], k: int) -> dict:
    dists = [group_kmer_distribution(ss, k) for ss in group_seqs]
    table = np.stack(dists)           # (G, 4**k)
    # keep only non-empty columns
    keep = table.sum(axis=0) > 0
    table = table[:, keep]
    chi2, p, dof, _ = chi2_contingency(table)
    # pairwise JSD on normalised distributions
    norm = table / table.sum(axis=1, keepdims=True)
    G = len(labels)
    jsd = np.zeros((G, G))
    for i in range(G):
        for j in range(G):
            jsd[i, j] = jensen_shannon(norm[i], norm[j])
    return {"chi2": float(chi2), "p": float(p), "dof": int(dof),
            "jsd": jsd, "labels": labels}


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_samples", type=int, default=4000)
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--mask_threshold", type=float, default=0.5)
    ap.add_argument("--meme_file", default=None,
                    help="Optional JASPAR-format MEME file for PWM motif scan.")
    ap.add_argument("--motif_score_threshold", type=float, default=0.85)
    ap.add_argument("--pca_out_dim", type=int, default=2)
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
    print(f"[ckpt] loaded {args.ckpt}  "
          f"keys={list(ckpt.keys()) if isinstance(ckpt, dict) else 'state_dict'}")

    # Use stored env centroids when available (from training), else refit here.
    centroids = None
    if isinstance(ckpt, dict) and ckpt.get("env_centroids") is not None:
        centroids = ckpt["env_centroids"].to(device)
        print(f"[env] using stored centroids with shape {tuple(centroids.shape)}")

    de_cfg = config.get("masked_separator", config.get("dual_encoder", {}))
    num_envs = de_cfg.get("num_envs", 3)

    groups = extract_groups(
        model, loader, device,
        num_samples=args.num_samples,
        threshold=args.mask_threshold,
        num_envs=num_envs,
        centroids=centroids,
    )

    inv_seqs = groups["invariant_seqs"]
    env_seqs = groups["env_seqs"]
    env_counts = [len(s) for s in env_seqs]
    print(f"[groups] invariant={len(inv_seqs)}  env_sizes={env_counts}  "
          f"mean mask activation={groups['hard_mask_mean']:.3f}")

    labels = ["invariant"] + [f"env{k}" for k in range(num_envs)]
    all_seqs = [inv_seqs] + env_seqs

    # Per-sample stats
    dfs = [per_sample_stats(s) for s in all_seqs]
    for lab, df in zip(labels, dfs):
        df.insert(0, "group", lab)
        df.to_csv(os.path.join(args.out_dir, f"stats_{lab}.csv"), index=False)

    # Aggregate scalar stats
    scalar_cols = ["length", "gc", "A", "C", "G", "T", "entropy", "low_complexity"]
    summary = pd.concat([df[["group"] + scalar_cols] for df in dfs], axis=0)
    summary_mean = summary.groupby("group").mean(numeric_only=True)
    summary_std = summary.groupby("group").std(numeric_only=True)
    summary_mean.to_csv(os.path.join(args.out_dir, "summary_mean.csv"))
    summary_std.to_csv(os.path.join(args.out_dir, "summary_std.csv"))
    print("\n[summary mean]\n", summary_mean.round(4))

    # Hypothesis tests on scalar stats
    test_df = run_tests(labels, dfs, scalar_cols)
    test_df.to_csv(os.path.join(args.out_dir, "tests_scalar.csv"), index=False)
    print("\n[kruskal across groups]\n",
          test_df[test_df.test == "kruskal"][["metric", "stat", "p"]].round(4))

    # k-mer distributional tests (3-mer, 6-mer) + JSD
    kmer_report = {}
    for k in (3, 6):
        res = kmer_chi2_across_groups(all_seqs, labels, k)
        kmer_report[f"{k}-mer"] = {
            "chi2": res["chi2"], "p": res["p"], "dof": res["dof"],
            "jsd_matrix": res["jsd"].tolist(),
            "labels": res["labels"],
        }
        pd.DataFrame(res["jsd"], index=labels, columns=labels).to_csv(
            os.path.join(args.out_dir, f"jsd_{k}mer.csv"))
    with open(os.path.join(args.out_dir, "kmer_tests.json"), "w") as f:
        json.dump(kmer_report, f, indent=2)
    print("\n[k-mer JSD 3-mer]")
    print(pd.DataFrame(kmer_report["3-mer"]["jsd_matrix"],
                       index=labels, columns=labels).round(4))

    # PCA on per-sample 3-mer vectors
    kmer_cols = [c for c in dfs[0].columns if c.startswith("kmer3_")]
    X = pd.concat(dfs, axis=0)[kmer_cols].values
    g = pd.concat(dfs, axis=0)["group"].values
    pca = PCA(n_components=args.pca_out_dim, random_state=0).fit(X)
    Z = pca.transform(X)
    pca_df = pd.DataFrame(Z, columns=[f"PC{i+1}" for i in range(args.pca_out_dim)])
    pca_df["group"] = g
    pca_df.to_csv(os.path.join(args.out_dir, "pca_3mer.csv"), index=False)
    print(f"[pca] explained var ratio = "
          f"{np.round(pca.explained_variance_ratio_, 4).tolist()}")

    # Optional motif enrichment
    if args.meme_file:
        pwms = parse_meme_pwms(args.meme_file)
        print(f"[motif] parsed {len(pwms)} PWMs from {args.meme_file}")
        if pwms:
            motif_rows = []
            for lab, seqs in zip(labels, all_seqs):
                hits = scan_motif_hits(seqs, pwms, threshold=args.motif_score_threshold)
                per_seq_total = hits.sum(axis=1)               # (N,)
                motif_rows.append({
                    "group": lab,
                    "n_seqs": len(seqs),
                    "mean_hits_per_seq": float(per_seq_total.mean()) if len(seqs) else np.nan,
                    "frac_seqs_with_any_hit": float((per_seq_total > 0).mean()) if len(seqs) else np.nan,
                })
                # per-motif hit rates
                per_motif = (hits > 0).mean(axis=0)
                np.savez(os.path.join(args.out_dir, f"motif_hits_{lab}.npz"),
                         hits=hits, per_motif_rate=per_motif,
                         names=np.array([m[0] for m in pwms]))
            pd.DataFrame(motif_rows).to_csv(
                os.path.join(args.out_dir, "motif_summary.csv"), index=False)
            print("\n[motif summary]")
            print(pd.DataFrame(motif_rows).round(4))

    # Dump final config/metadata
    with open(os.path.join(args.out_dir, "run_meta.json"), "w") as f:
        json.dump({
            "ckpt": args.ckpt,
            "split": args.split,
            "num_samples_requested": args.num_samples,
            "num_invariant_seqs": len(inv_seqs),
            "env_group_sizes": env_counts,
            "mask_threshold": args.mask_threshold,
            "mean_mask_activation": float(groups["hard_mask_mean"]),
            "num_envs": num_envs,
        }, f, indent=2)
    print(f"\n[done] outputs written to {args.out_dir}")


if __name__ == "__main__":
    main()
