"""
Entry point for the proposal's guided generation pipeline.

Usage:
    # 1. train ensemble property predictor
    python -m scripts.main_guided --config configs/enhancer_gosai_guided.yaml \
                                  --out_dir ./runs/ensemble --train_ensemble

    # 2. guided generation with frozen Dirichlet FM backbone
    python -m scripts.main_guided --config configs/enhancer_gosai_guided.yaml \
                                  --out_dir ./runs/guided --generate
"""

import argparse
import yaml
from ml_collections.config_dict import ConfigDict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--num_batches", type=int, default=2)
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--train_ensemble", action="store_true")
    g.add_argument("--generate",       action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = ConfigDict(yaml.safe_load(f))
    config.out_dir = args.out_dir

    if args.train_ensemble:
        from scripts.property_scorer_trainer import EnsembleTrainer
        EnsembleTrainer(config).train()

    elif args.generate:
        import os, csv
        from scripts.guided_sampler import GuidedSampler, INT_TO_BASE

        sampler = GuidedSampler(config)
        B = config.loader.eval_batch_size
        L = config.dataset.seq_length

        os.makedirs(args.out_dir, exist_ok=True)
        out_path = os.path.join(args.out_dir, "guided_sequences.csv")
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["seq", "mu", "sigma", "score"])
            for _ in range(args.num_batches):
                seqs, _ = sampler.sample(B, L)
                ranked = sampler.score_and_rank(seqs)
                for j in range(seqs.size(0)):
                    writer.writerow([
                        "".join(INT_TO_BASE[t] for t in seqs[j].tolist()),
                        f"{ranked['mu'][j].item():.4f}",
                        f"{ranked['sigma'][j].item():.4f}",
                        f"{ranked['score'][j].item():.4f}",
                    ])
        print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
