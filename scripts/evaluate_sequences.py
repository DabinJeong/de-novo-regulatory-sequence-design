"""
Evaluation metrics for generated regulatory DNA sequences.

Computes a standardised set of metrics so that generated sequences from
any method (guided sampler, DRAKES, SVDD, Ctrl-DNA, DNA-Diffusion) can
be compared on equal footing.

Metrics
-------
Property (require PropertyScorer / EnsembleRegressor checkpoint):
    - mu_hat       : predicted activity (mean, median, top-k mean)
    - sigma_hat    : epistemic uncertainty (mean)
    - score        : mu - gamma * sigma (mean, median, top-k mean)

Diversity & novelty (sequence-level):
    - novelty      : min Hamming distance from each generated seq to
                     the closest training sequence (mean, median)
    - diversity    : mean pairwise Hamming distance among generated seqs
    - unique_ratio : fraction of unique sequences

Distributional:
    - kmer_kl      : KL(generated || training) of k-mer frequency spectrum
    - gc_mean      : mean GC content of generated sequences
    - gc_std       : std  GC content
    - gc_kl        : KL(gc_gen || gc_train) discretised to 20 bins

CLI
---
    python -m scripts.evaluate_sequences \
        --generated  runs/guided/guided_sequences.csv \
        --train_data data/gosai_all.csv \
        --scorer_ckpt runs/ensemble/ensemble_best.ckpt \
        --out_dir     runs/guided/eval \
        --k 5         # k for k-mer spectrum \
        --top_n 100   # top-N for top-k metrics
"""

import argparse
import csv
import json
import os
from collections import Counter

import numpy as np
import torch

BASE_TO_INT = {"A": 0, "C": 1, "G": 2, "T": 3}
INT_TO_BASE = {0: "A", 1: "C", 2: "G", 3: "T"}


# ---------------------------------------------------------------------------
# Sequence I/O
# ---------------------------------------------------------------------------
def load_sequences_csv(path: str, seq_col: str = "seq"):
    seqs = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            seqs.append(row[seq_col].upper())
    return seqs


def load_train_sequences(path: str, seq_col: str = "seq", max_seqs: int = None):
    seqs = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_seqs and i >= max_seqs:
                break
            seqs.append(row[seq_col].upper())
    return seqs


def seqs_to_tensor(seqs):
    return torch.tensor(
        [[BASE_TO_INT.get(c, 0) for c in s] for s in seqs], dtype=torch.long
    )


# ---------------------------------------------------------------------------
# Property metrics (mu, sigma, score)
# ---------------------------------------------------------------------------
def compute_property_metrics(gen_tensor, scorer, gamma: float, top_n: int,
                             device: torch.device, batch_size: int = 256):
    from sequence_generation.model.ensemble_regressor import EnsembleRegressor

    mus, sigmas = [], []
    for i in range(0, gen_tensor.size(0), batch_size):
        chunk = gen_tensor[i:i + batch_size].to(device)
        mu, var = scorer.mu_sigma2({"seqs": chunk})
        mus.append(mu.squeeze(-1).cpu())
        sigmas.append(var.clamp_min(1e-12).sqrt().squeeze(-1).cpu())
    mu = torch.cat(mus, 0).numpy()
    sigma = torch.cat(sigmas, 0).numpy()
    score = mu - gamma * sigma

    top_n = min(top_n, len(score))
    top_idx = np.argsort(score)[-top_n:]

    return {
        "mu_mean": float(mu.mean()),
        "mu_median": float(np.median(mu)),
        "mu_top_mean": float(mu[top_idx].mean()),
        "sigma_mean": float(sigma.mean()),
        "sigma_median": float(np.median(sigma)),
        "score_mean": float(score.mean()),
        "score_median": float(np.median(score)),
        "score_top_mean": float(score[top_idx].mean()),
    }


# ---------------------------------------------------------------------------
# Diversity & novelty
# ---------------------------------------------------------------------------
def hamming_distance_matrix(seqs_a, seqs_b, batch_size: int = 512):
    """Returns (len_a, len_b) numpy array of normalised Hamming distances."""
    a = seqs_to_tensor(seqs_a)
    b = seqs_to_tensor(seqs_b)
    L = a.size(1)
    dists = np.zeros((len(seqs_a), len(seqs_b)), dtype=np.float32)
    for i in range(0, len(seqs_a), batch_size):
        a_chunk = a[i:i + batch_size].unsqueeze(1)  # (chunk, 1, L)
        d = (a_chunk != b.unsqueeze(0)).float().mean(-1)  # (chunk, len_b)
        dists[i:i + batch_size] = d.numpy()
    return dists


def compute_novelty(gen_seqs, train_seqs, max_train: int = 10000):
    train_sub = train_seqs[:max_train]
    dists = hamming_distance_matrix(gen_seqs, train_sub)
    min_dists = dists.min(axis=1)
    return {
        "novelty_mean": float(min_dists.mean()),
        "novelty_median": float(np.median(min_dists)),
        "novelty_min": float(min_dists.min()),
        "novelty_max": float(min_dists.max()),
    }


def compute_diversity(gen_seqs, max_seqs: int = 2000):
    sub = gen_seqs[:max_seqs]
    t = seqs_to_tensor(sub)
    N = len(sub)
    total = 0.0
    count = 0
    for i in range(0, N, 256):
        chunk = t[i:i + 256].unsqueeze(1)
        d = (chunk != t.unsqueeze(0)).float().mean(-1)
        mask = torch.triu(torch.ones(chunk.size(0), N, dtype=torch.bool), diagonal=i + 1)
        if i > 0:
            mask[:, :i] = False
        total += d[mask[:chunk.size(0)]].sum().item()
        count += mask[:chunk.size(0)].sum().item()
    return {
        "diversity": float(total / max(count, 1)),
        "unique_ratio": float(len(set(gen_seqs)) / len(gen_seqs)),
        "num_sequences": len(gen_seqs),
    }


# ---------------------------------------------------------------------------
# k-mer spectrum
# ---------------------------------------------------------------------------
def kmer_frequencies(seqs, k: int = 5):
    counts = Counter()
    total = 0
    for s in seqs:
        for i in range(len(s) - k + 1):
            kmer = s[i:i + k]
            if all(c in "ACGT" for c in kmer):
                counts[kmer] += 1
                total += 1
    freq = {kmer: c / total for kmer, c in counts.items()} if total > 0 else {}
    return freq


def kl_divergence(p_freq, q_freq, pseudocount: float = 1e-8):
    all_kmers = set(p_freq) | set(q_freq)
    total_p = sum(p_freq.values()) + pseudocount * len(all_kmers)
    total_q = sum(q_freq.values()) + pseudocount * len(all_kmers)
    kl = 0.0
    for kmer in all_kmers:
        p = (p_freq.get(kmer, 0) + pseudocount) / total_p
        q = (q_freq.get(kmer, 0) + pseudocount) / total_q
        kl += p * np.log(p / q)
    return float(kl)


def compute_kmer_metrics(gen_seqs, train_seqs, k: int = 5):
    gen_freq = kmer_frequencies(gen_seqs, k)
    train_freq = kmer_frequencies(train_seqs, k)
    return {
        f"{k}mer_kl": kl_divergence(gen_freq, train_freq),
        f"{k}mer_unique_gen": len(gen_freq),
        f"{k}mer_unique_train": len(train_freq),
    }


# ---------------------------------------------------------------------------
# GC content
# ---------------------------------------------------------------------------
def gc_content(seq: str) -> float:
    gc = sum(1 for c in seq if c in "GC")
    return gc / len(seq) if len(seq) > 0 else 0.0


def gc_kl(gen_seqs, train_seqs, n_bins: int = 20):
    gen_gc = [gc_content(s) for s in gen_seqs]
    train_gc = [gc_content(s) for s in train_seqs]
    bins = np.linspace(0, 1, n_bins + 1)
    gen_hist = np.histogram(gen_gc, bins=bins, density=True)[0] + 1e-8
    train_hist = np.histogram(train_gc, bins=bins, density=True)[0] + 1e-8
    gen_hist /= gen_hist.sum()
    train_hist /= train_hist.sum()
    kl = float((gen_hist * np.log(gen_hist / train_hist)).sum())
    return {
        "gc_mean": float(np.mean(gen_gc)),
        "gc_std": float(np.std(gen_gc)),
        "gc_kl": kl,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", type=str, required=True,
                        help="CSV with generated sequences (must have 'seq' column)")
    parser.add_argument("--train_data", type=str, default=None,
                        help="CSV with training sequences for novelty/kmer metrics")
    parser.add_argument("--scorer_ckpt", type=str, default=None,
                        help="PropertyScorer checkpoint for mu/sigma metrics")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--k", type=int, default=5, help="k for k-mer spectrum")
    parser.add_argument("--top_n", type=int, default=100, help="top-N for top-k metrics")
    parser.add_argument("--gamma", type=float, default=1.0, help="gamma for score = mu - gamma*sigma")
    parser.add_argument("--seq_col", type=str, default="seq")
    parser.add_argument("--max_train", type=int, default=10000,
                        help="Cap training seqs for novelty computation")
    args = parser.parse_args()

    print(f"[eval] loading generated sequences from {args.generated}")
    gen_seqs = load_sequences_csv(args.generated, args.seq_col)
    print(f"[eval] {len(gen_seqs)} generated sequences loaded")

    results = {}

    # -- Property metrics --
    if args.scorer_ckpt and os.path.exists(args.scorer_ckpt):
        from sequence_generation.model.ensemble_regressor import EnsembleRegressor
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        scorer = EnsembleRegressor(alphabet_size=4)
        sd = torch.load(args.scorer_ckpt, map_location="cpu")
        scorer.load_state_dict(sd.get("model", sd))
        scorer.to(device).eval()
        for p in scorer.parameters():
            p.requires_grad_(False)

        gen_tensor = seqs_to_tensor(gen_seqs)
        prop = compute_property_metrics(gen_tensor, scorer, args.gamma,
                                        args.top_n, device)
        results["property"] = prop
        print(f"[eval] property: mu_mean={prop['mu_mean']:.4f}  "
              f"score_mean={prop['score_mean']:.4f}  "
              f"score_top{args.top_n}={prop['score_top_mean']:.4f}")
    else:
        print("[eval] no scorer checkpoint — skipping property metrics")

    # -- Diversity --
    div = compute_diversity(gen_seqs)
    results["diversity"] = div
    print(f"[eval] diversity={div['diversity']:.4f}  "
          f"unique_ratio={div['unique_ratio']:.4f}")

    # -- Novelty & distributional (need training data) --
    if args.train_data and os.path.exists(args.train_data):
        train_seqs = load_train_sequences(args.train_data, args.seq_col,
                                          max_seqs=args.max_train)
        print(f"[eval] {len(train_seqs)} training sequences loaded")

        nov = compute_novelty(gen_seqs, train_seqs, max_train=args.max_train)
        results["novelty"] = nov
        print(f"[eval] novelty_mean={nov['novelty_mean']:.4f}  "
              f"novelty_median={nov['novelty_median']:.4f}")

        kmer = compute_kmer_metrics(gen_seqs, train_seqs, k=args.k)
        results["kmer"] = kmer
        print(f"[eval] {args.k}mer_kl={kmer[f'{args.k}mer_kl']:.6f}")

        gc = gc_kl(gen_seqs, train_seqs)
        results["gc"] = gc
        print(f"[eval] gc_mean={gc['gc_mean']:.4f}  gc_kl={gc['gc_kl']:.6f}")
    else:
        gc = gc_kl(gen_seqs, [])
        results["gc"] = {"gc_mean": gc["gc_mean"], "gc_std": gc["gc_std"]}
        print(f"[eval] gc_mean={gc['gc_mean']:.4f}  (no train data for kmer/novelty)")

    # -- Save --
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        out_path = os.path.join(args.out_dir, "eval_metrics.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[eval] saved -> {out_path}")
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
