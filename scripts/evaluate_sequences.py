"""
Evaluation metrics for generated regulatory DNA sequences.

Computes a standardised set of metrics so that generated sequences from
any method (guided sampler, DRAKES, SVDD, Ctrl-DNA, DNA-Diffusion) can
be compared on equal footing.

Metrics
-------
Property (our scorer; circular — diagnostic / reward-hacking check):
    - mu_hat       : predicted activity (mean, median, top-k mean)
    - sigma_hat    : epistemic uncertainty (mean)
    - score        : mu - gamma * sigma (mean, median, top-k mean)

Independent oracle (external activity judge; require --oracle_ckpt):
    - independent_mu : DRAKES reward_oracle_eval.ckpt (grelu Enformer, 3-task),
                       non-circular. Also reports Pearson vs our scorer.

Sequence-level:
    - diversity    : mean pairwise Hamming among generated seqs
    - unique_ratio : fraction of unique sequences
    - novelty      : min Hamming from each gen seq to nearest training seq
                     (goal: ↑, far from training)

Composition (gen-only, descriptive):
    - gc_mean/std  : GC-content summary — extreme values flag implausibility

Feature-space OOD (PropertyScorer penultimate, 128-D; goal: ↑ vs training):
    - mmd_rbf           : MMD^2 with RBF kernel (median bandwidth)
    - sliced_wasserstein: mean 1D Wasserstein over random projections
                          (Rabin+2011, Bonneel+2015)
    - mu_wasserstein    : 1D Wasserstein on predicted-activity marginal

JASPAR motif (gen-only, biological plausibility; require --jaspar):
    - motif_hits_per_seq_mean / median : PSSM hits per sequence at FPR=1e-3
    - motif_coverage                   : fraction of seqs with >=1 hit

CLI
---
    python -m scripts.evaluate_sequences \
        --generated  runs/guided/guided_sequences.csv \
        --train_data data/gosai_all.csv \
        --scorer_ckpt runs/property_scorer/best.ckpt \
        --oracle_ckpt DRAKES_data/.../reward_oracle_eval.ckpt \
        --oracle_target_idx 0 \
        --out_dir     runs/guided/eval \
        --top_n 100          # top-N for top-k metrics \
        --n_projections 100  # random projections for sliced Wasserstein
"""

import argparse
import csv
import json
import os

import numpy as np
import torch

BASE_TO_INT = {"A": 0, "C": 1, "G": 2, "T": 3}
INT_TO_BASE = {0: "A", 1: "C", 2: "G", 3: "T"}


# ---------------------------------------------------------------------------
# Sequence I/O
# ---------------------------------------------------------------------------
def load_sequences_csv(path: str, seq_col: str = "seq",
                       cell_type: str = None,
                       cell_type_col: str = "cell_type"):
    """Load sequences from CSV, optionally filtering by cell_type value.

    If `cell_type` is set, only rows whose `cell_type_col` matches are kept.
    Matching is case-insensitive substring (so 'HepG2' matches
    'HepG2_ENCLB441ZZZ' from DNA-Diffusion's tag naming).
    """
    seqs = []
    seen_any_match_col = False
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if cell_type is not None:
                if cell_type_col not in row:
                    continue
                seen_any_match_col = True
                if cell_type.lower() not in row[cell_type_col].lower():
                    continue
            seqs.append(row[seq_col].upper())
    if cell_type is not None and not seen_any_match_col:
        raise ValueError(
            f"--cell_type={cell_type!r} requested but column {cell_type_col!r} "
            f"not present in {path}. Drop the flag or use a CSV with that column."
        )
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
def score_tensor(tensor, scorer, device, batch_size: int = 256):
    """Run PropertyScorer over sequences, return (mu, sigma) as numpy arrays."""
    mus, sigmas = [], []
    with torch.no_grad():
        for i in range(0, tensor.size(0), batch_size):
            chunk = tensor[i:i + batch_size].to(device)
            mu, var = scorer.mu_sigma2({"seqs": chunk})
            mus.append(mu.squeeze(-1).cpu())
            sigmas.append(var.clamp_min(1e-12).sqrt().squeeze(-1).cpu())
    return torch.cat(mus, 0).numpy(), torch.cat(sigmas, 0).numpy()


def compute_property_metrics(mu, sigma, gamma: float, top_n: int):
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
# Independent oracle (DRAKES-distributed reward_oracle_eval.ckpt, grelu-based)
# ---------------------------------------------------------------------------
def load_independent_oracle(ckpt_path: str, device):
    """Load the DRAKES-distributed grelu LightningModel used as external oracle.

    The checkpoint was saved via grelu's LightningModel wrapper around
    EnformerPretrainedModel (3-task regression: hepg2, k562, sknsh).
    """
    # DRAKES' ckpt nests data_params/model_params/train_params inside
    # hyper_parameters, but newer grelu expects them at the top level and
    # also reads `performance` directly. Patch before the import.
    from sequence_generation.utils.grelu_compat import patch_grelu_lightning_compat
    patch_grelu_lightning_compat()
    from grelu.lightning import LightningModel
    model = LightningModel.load_from_checkpoint(ckpt_path, map_location=device)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _seqs_to_onehot(tensor: torch.Tensor) -> torch.Tensor:
    """(B, L) int -> (B, 4, L) float one-hot, the layout grelu models expect."""
    return torch.nn.functional.one_hot(tensor, num_classes=4).permute(0, 2, 1).float()


def score_with_oracle(tensor: torch.Tensor, oracle, device, target_idx: int,
                      batch_size: int = 128) -> np.ndarray:
    """Run the independent oracle over (B, L) token IDs and return (N,) predictions
    for the requested target_idx column (0=hepg2, 1=k562, 2=sknsh)."""
    preds = []
    with torch.no_grad():
        for i in range(0, tensor.size(0), batch_size):
            chunk = _seqs_to_onehot(tensor[i:i + batch_size].to(device))
            out = oracle(chunk)
            if out.dim() == 3:
                out = out.mean(dim=-1)
            preds.append(out[:, target_idx].detach().cpu())
    return torch.cat(preds, 0).numpy()


def compute_independent_oracle_metrics(mu_indep: np.ndarray, top_n: int,
                                       mu_ours: np.ndarray = None):
    top_n = min(top_n, len(mu_indep))
    top_idx = np.argsort(mu_indep)[-top_n:]
    metrics = {
        "independent_mu_mean": float(mu_indep.mean()),
        "independent_mu_median": float(np.median(mu_indep)),
        "independent_mu_top_mean": float(mu_indep[top_idx].mean()),
    }
    if mu_ours is not None and len(mu_ours) == len(mu_indep):
        corr = float(np.corrcoef(mu_indep, mu_ours)[0, 1])
        metrics["pearson_vs_ours"] = corr
    return metrics


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
# JASPAR motif scanning
#
# Loads vertebrate core PWMs via pyjaspar, builds biopython PSSMs, scans each
# sequence on both strands at a fixed FPR threshold, and reports:
#   - motif_hits_per_seq_mean/median : hit density per sequence
#   - motif_coverage                 : fraction of seqs with >=1 hit
# ---------------------------------------------------------------------------
def _load_jaspar_pssms(release: str, collection: str, tax_group: str,
                       pseudocount: float, max_motifs: int = None):
    """Return list of (motif_id, biopython PSSM) pairs."""
    from pyjaspar import jaspardb
    jdb = jaspardb(release=release)
    kwargs = {"collection": collection}
    if tax_group:
        kwargs["tax_group"] = [tax_group]
    motifs = jdb.fetch_motifs(**kwargs)
    if max_motifs is not None:
        motifs = motifs[:max_motifs]

    pssms = []
    for m in motifs:
        pwm = m.counts.normalize(pseudocounts=pseudocount)
        pssm = pwm.log_odds()
        pssms.append((m.matrix_id, pssm))
    return pssms


def _pssm_thresholds(pssms, background, fpr: float):
    """Compute per-motif score threshold at the given false-positive rate."""
    thresholds = []
    for _, pssm in pssms:
        dist = pssm.distribution(background=background, precision=10**3)
        thresholds.append(dist.threshold_fpr(fpr))
    return thresholds


def scan_jaspar_motifs(seqs, release: str = "JASPAR2022",
                       collection: str = "CORE",
                       tax_group: str = "vertebrates",
                       fpr: float = 1e-3, pseudocount: float = 0.5,
                       max_motifs: int = None):
    """Scan each sequence on both strands against JASPAR PSSMs.

    Returns:
        hits_per_seq : (N,) int — total hit count per sequence (any motif)
        motif_freq   : (M,) float — per-motif hit rate averaged over sequences
        motif_ids    : list[str] of length M
    """
    from Bio.Seq import Seq

    pssms = _load_jaspar_pssms(release, collection, tax_group,
                               pseudocount, max_motifs)
    background = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}
    thresholds = _pssm_thresholds(pssms, background, fpr)

    N, M = len(seqs), len(pssms)
    hits_per_seq = np.zeros(N, dtype=np.int64)
    motif_hits = np.zeros(M, dtype=np.int64)

    for i, s in enumerate(seqs):
        bseq = Seq(s)
        for j, (_, pssm) in enumerate(pssms):
            # `both=True` scans reverse-complement too — regulatory motifs
            # can bind either strand at enhancer elements.
            count = sum(1 for _ in pssm.search(bseq, threshold=thresholds[j],
                                               both=True))
            hits_per_seq[i] += count
            motif_hits[j] += count

    motif_freq = motif_hits.astype(np.float64) / max(N, 1)
    return hits_per_seq, motif_freq, [mid for mid, _ in pssms]


def compute_jaspar_metrics(gen_seqs, release: str = "JASPAR2022",
                           collection: str = "CORE",
                           tax_group: str = "vertebrates",
                           fpr: float = 1e-3, max_motifs: int = None):
    gen_hits, _, motif_ids = scan_jaspar_motifs(
        gen_seqs, release=release, collection=collection,
        tax_group=tax_group, fpr=fpr, max_motifs=max_motifs)

    return {
        "motif_hits_per_seq_mean": float(gen_hits.mean()),
        "motif_hits_per_seq_median": float(np.median(gen_hits)),
        "motif_coverage": float((gen_hits > 0).mean()),
        "num_motifs": len(motif_ids),
        "fpr_threshold": fpr,
        "release": release,
        "collection": collection,
    }


# ---------------------------------------------------------------------------
# Feature-space distribution metrics (MMD, Sliced Wasserstein)
#
# Goal: ↑ distance from training in PropertyScorer's penultimate feature
# space — our objective is to generate OOD sequences. These metrics should
# be read jointly with the independent oracle (OOD + high oracle μ = real
# functional novelty; OOD + low oracle μ = reward hacking).
# ---------------------------------------------------------------------------
def extract_penultimate_features(tensor, scorer, device, batch_size: int = 256,
                                 member_idx: int = 0):
    """Capture the input to final_linear from one PropertyScorer member.

    Member 0 is used by default; using one member (vs all) keeps feature
    dimensionality low and avoids correlated redundancy across members.
    """
    member = scorer.members[member_idx]
    feats = []

    def pre_hook(_module, inputs):
        feats.append(inputs[0].detach().cpu())

    handle = member.final_linear.register_forward_pre_hook(pre_hook)
    try:
        with torch.no_grad():
            for i in range(0, tensor.size(0), batch_size):
                chunk = tensor[i:i + batch_size].to(device)
                _ = member({"seqs": chunk})
    finally:
        handle.remove()
    return torch.cat(feats, 0).numpy()


def _sq_dists(A, B):
    A2 = (A ** 2).sum(axis=1, keepdims=True)
    B2 = (B ** 2).sum(axis=1, keepdims=True).T
    return np.maximum(A2 + B2 - 2.0 * A @ B.T, 0.0)


def mmd_rbf(X, Y, bandwidth: float = None):
    """Unbiased MMD^2 with RBF kernel; median heuristic bandwidth if None."""
    XX = _sq_dists(X, X)
    YY = _sq_dists(Y, Y)
    XY = _sq_dists(X, Y)

    if bandwidth is None:
        nz = XY[XY > 0]
        bandwidth = float(np.median(np.sqrt(nz))) if nz.size else 1.0
        bandwidth = max(bandwidth, 1e-6)

    gamma = 1.0 / (2.0 * bandwidth * bandwidth)
    Kxx = np.exp(-gamma * XX)
    Kyy = np.exp(-gamma * YY)
    Kxy = np.exp(-gamma * XY)

    n, m = X.shape[0], Y.shape[0]
    mmd2 = (
        (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1))
        + (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1))
        - 2.0 * Kxy.sum() / (n * m)
    )
    return float(max(mmd2, 0.0)), float(bandwidth)


def sliced_wasserstein(X, Y, n_projections: int = 100, seed: int = 0):
    from scipy.stats import wasserstein_distance

    rng = np.random.default_rng(seed)
    D = X.shape[1]
    projs = rng.standard_normal(size=(n_projections, D))
    projs /= np.linalg.norm(projs, axis=1, keepdims=True) + 1e-12

    X_proj = X @ projs.T
    Y_proj = Y @ projs.T

    return float(np.mean([
        wasserstein_distance(X_proj[:, i], Y_proj[:, i])
        for i in range(n_projections)
    ]))


def compute_feature_distribution_metrics(gen_tensor, train_tensor, scorer, device,
                                         max_train_feats: int = 5000,
                                         n_projections: int = 100):
    gen_feats = extract_penultimate_features(gen_tensor, scorer, device)

    train_sub = train_tensor
    if train_tensor.size(0) > max_train_feats:
        idx = np.random.default_rng(0).choice(
            train_tensor.size(0), size=max_train_feats, replace=False)
        train_sub = train_tensor[idx]
    train_feats = extract_penultimate_features(train_sub, scorer, device)

    mmd2, bandwidth = mmd_rbf(gen_feats, train_feats)
    sw = sliced_wasserstein(gen_feats, train_feats, n_projections=n_projections)

    return {
        "mmd_rbf": mmd2,
        "mmd_bandwidth": bandwidth,
        "sliced_wasserstein": sw,
        "feature_dim": int(gen_feats.shape[1]),
        "n_gen_feat": int(gen_feats.shape[0]),
        "n_train_feat": int(train_feats.shape[0]),
        "n_projections": n_projections,
    }


def wasserstein_1d(x, y):
    from scipy.stats import wasserstein_distance
    return float(wasserstein_distance(np.asarray(x), np.asarray(y)))


# ---------------------------------------------------------------------------
# GC content
# ---------------------------------------------------------------------------
def gc_content(seq: str) -> float:
    gc = sum(1 for c in seq if c in "GC")
    return gc / len(seq) if len(seq) > 0 else 0.0


def compute_gc_stats(gen_seqs):
    gen_gc = [gc_content(s) for s in gen_seqs]
    return {
        "gc_mean": float(np.mean(gen_gc)),
        "gc_std": float(np.std(gen_gc)),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", type=str, required=True,
                        help="CSV with generated sequences (must have 'seq' column)")
    parser.add_argument("--train_data", type=str, default=None,
                        help="CSV with training sequences for novelty + feature-space metrics")
    parser.add_argument("--scorer_ckpt", type=str, default=None,
                        help="PropertyScorer checkpoint for mu/sigma metrics")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--top_n", type=int, default=100, help="top-N for top-k metrics")
    parser.add_argument("--gamma", type=float, default=1.0, help="gamma for score = mu - gamma*sigma")
    parser.add_argument("--seq_col", type=str, default="seq")
    parser.add_argument("--max_train", type=int, default=10000,
                        help="Cap training seqs for novelty computation")
    parser.add_argument("--max_train_feats", type=int, default=5000,
                        help="Cap training seqs for feature-space MMD/Wasserstein")
    parser.add_argument("--n_projections", type=int, default=100,
                        help="Random projections for sliced Wasserstein")
    parser.add_argument("--cell_type", type=str, default=None,
                        help="If set, keep only generated rows whose 'cell_type' "
                             "column contains this string (case-insensitive). "
                             "Useful for filtering DNA-Diffusion's per-cell-type "
                             "output down to e.g. HepG2 only.")
    parser.add_argument("--oracle_ckpt", type=str, default=None,
                        help="Path to DRAKES-distributed reward_oracle_eval.ckpt "
                             "(grelu Enformer). Enables independent oracle metrics.")
    parser.add_argument("--oracle_target_idx", type=int, default=0,
                        help="Which of the oracle's 3 heads to report "
                             "(0=hepg2, 1=k562, 2=sknsh). Default 0.")
    parser.add_argument("--oracle_batch_size", type=int, default=128)
    parser.add_argument("--jaspar", action="store_true",
                        help="Enable JASPAR motif scanning metrics. Requires "
                             "pyjaspar and biopython.")
    parser.add_argument("--jaspar_release", type=str, default="JASPAR2022")
    parser.add_argument("--jaspar_collection", type=str, default="CORE")
    parser.add_argument("--jaspar_tax_group", type=str, default="vertebrates")
    parser.add_argument("--jaspar_fpr", type=float, default=1e-3,
                        help="Per-PSSM false-positive rate for hit threshold.")
    parser.add_argument("--jaspar_max_motifs", type=int, default=None,
                        help="Cap the number of JASPAR motifs scanned (useful "
                             "for smoke tests; unset = scan all).")
    args = parser.parse_args()

    print(f"[eval] loading generated sequences from {args.generated}")
    gen_seqs = load_sequences_csv(args.generated, args.seq_col,
                                  cell_type=args.cell_type)
    if args.cell_type:
        print(f"[eval] filtered to cell_type~={args.cell_type!r}")
    print(f"[eval] {len(gen_seqs)} generated sequences loaded")

    results = {}
    scorer = None
    device = None
    gen_tensor = None
    gen_mu = None

    # -- Property metrics --
    if args.scorer_ckpt and os.path.exists(args.scorer_ckpt):
        from sequence_generation.model.property_scorer import PropertyScorer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        scorer = PropertyScorer(alphabet_size=4)
        sd = torch.load(args.scorer_ckpt, map_location="cpu")
        scorer.load_state_dict(sd.get("model", sd))
        scorer.to(device).eval()
        for p in scorer.parameters():
            p.requires_grad_(False)

        gen_tensor = seqs_to_tensor(gen_seqs)
        gen_mu, gen_sigma = score_tensor(gen_tensor, scorer, device)
        prop = compute_property_metrics(gen_mu, gen_sigma, args.gamma, args.top_n)
        results["property"] = prop
        print(f"[eval] property: mu_mean={prop['mu_mean']:.4f}  "
              f"score_mean={prop['score_mean']:.4f}  "
              f"score_top{args.top_n}={prop['score_top_mean']:.4f}")
    else:
        print("[eval] no scorer checkpoint — skipping property + feature metrics")

    # -- Independent oracle (DRAKES reward_oracle_eval) --
    if args.oracle_ckpt and os.path.exists(args.oracle_ckpt):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if gen_tensor is None:
            gen_tensor = seqs_to_tensor(gen_seqs)
        print(f"[eval] loading independent oracle from {args.oracle_ckpt}")
        oracle = load_independent_oracle(args.oracle_ckpt, device)
        mu_indep = score_with_oracle(gen_tensor, oracle, device,
                                     target_idx=args.oracle_target_idx,
                                     batch_size=args.oracle_batch_size)
        indep = compute_independent_oracle_metrics(
            mu_indep, args.top_n, mu_ours=gen_mu)
        results["independent_oracle"] = {
            **indep,
            "target_idx": args.oracle_target_idx,
            "ckpt_path": args.oracle_ckpt,
        }
        print(f"[eval] independent_mu_mean={indep['independent_mu_mean']:.4f}  "
              f"top{args.top_n}={indep['independent_mu_top_mean']:.4f}"
              + (f"  pearson_vs_ours={indep['pearson_vs_ours']:.4f}"
                 if 'pearson_vs_ours' in indep else ''))
        del oracle
        if device.type == "cuda":
            torch.cuda.empty_cache()
    elif args.oracle_ckpt:
        print(f"[eval] --oracle_ckpt {args.oracle_ckpt} not found — skipping")

    # -- Diversity (gen-only) --
    div = compute_diversity(gen_seqs)
    results["diversity"] = div
    print(f"[eval] diversity={div['diversity']:.4f}  "
          f"unique_ratio={div['unique_ratio']:.4f}")

    # -- GC content (gen-only, descriptive) --
    gc = compute_gc_stats(gen_seqs)
    results["gc"] = gc
    print(f"[eval] gc_mean={gc['gc_mean']:.4f}  gc_std={gc['gc_std']:.4f}")

    # -- JASPAR motif (gen-only, biological plausibility) --
    if args.jaspar:
        print(f"[eval] scanning JASPAR {args.jaspar_release}/"
              f"{args.jaspar_collection}/{args.jaspar_tax_group} "
              f"at FPR={args.jaspar_fpr}")
        jmetrics = compute_jaspar_metrics(
            gen_seqs,
            release=args.jaspar_release,
            collection=args.jaspar_collection,
            tax_group=args.jaspar_tax_group,
            fpr=args.jaspar_fpr,
            max_motifs=args.jaspar_max_motifs)
        results["jaspar"] = jmetrics
        print(f"[eval] motif_hits/seq={jmetrics['motif_hits_per_seq_mean']:.2f}  "
              f"coverage={jmetrics['motif_coverage']:.3f}  "
              f"motifs={jmetrics['num_motifs']}")

    # -- Metrics requiring training data --
    if args.train_data and os.path.exists(args.train_data):
        train_seqs = load_train_sequences(args.train_data, args.seq_col,
                                          max_seqs=args.max_train)
        print(f"[eval] {len(train_seqs)} training sequences loaded")

        # Seq-level novelty (Hamming to nearest training seq; goal: ↑)
        nov = compute_novelty(gen_seqs, train_seqs, max_train=args.max_train)
        results["novelty"] = nov
        print(f"[eval] novelty_mean={nov['novelty_mean']:.4f}  "
              f"novelty_median={nov['novelty_median']:.4f}")

        # Feature-space OOD (MMD / Sliced W / mu_wasserstein; goal: ↑)
        if scorer is not None:
            train_tensor = seqs_to_tensor(train_seqs)
            train_mu, _ = score_tensor(train_tensor, scorer, device)
            feat_dist = compute_feature_distribution_metrics(
                gen_tensor, train_tensor, scorer, device,
                max_train_feats=args.max_train_feats,
                n_projections=args.n_projections)
            feat_dist["mu_wasserstein"] = wasserstein_1d(gen_mu, train_mu)
            results["feature_dist"] = feat_dist
            print(f"[eval] mmd_rbf={feat_dist['mmd_rbf']:.6f}  "
                  f"sliced_wasserstein={feat_dist['sliced_wasserstein']:.4f}  "
                  f"mu_wasserstein={feat_dist['mu_wasserstein']:.4f}")

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
