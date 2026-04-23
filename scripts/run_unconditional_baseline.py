"""
Unconditional DirichletFM baseline — anchor for ablation.

No guidance, no property pull, no separator. Just draws from the frozen
DirichletFM backbone:

    x_0  ~ Dirichlet(1,...,1)
    x_t <- x_t + flow(x_t, t) * dt                # no guidance term
    seq  = argmax(logits_pred)

Writes a single-column CSV (`seq`) for downstream evaluation by
`scripts.evaluate_sequences`, which scores with the independent DRAKES
oracle (primary, non-circular) and PropertyScorer (circular,
diagnostic).

Usage:
    python -m scripts.run_unconditional_baseline \
      --config configs/enhancer_gosai_guided.yaml \
      --out_dir ./runs/unconditional \
      --num_batches 20
"""

import argparse
import csv
import os

import torch
import torch.nn.functional as F
import yaml
from ml_collections.config_dict import ConfigDict

from sequence_generation.utils.flow_utils import (
    DirichletConditionalFlow,
    expand_simplex,
)
from sequence_generation.utils.train_utils import load_generator


INT_TO_BASE = {0: "A", 1: "C", 2: "G", 3: "T"}


class UnconditionalSampler:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model, _, _ = load_generator(config)
        self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.K = config.model.alphabet_size
        self.condflow = DirichletConditionalFlow(
            K=self.K,
            alpha_spacing=0.01,
            alpha_max=config.model.alpha_max,
        )

        s = config.sampling
        self.n_steps = s.get("n_steps", 128)
        self.flow_temp = s.get("flow_temp", 1.0)
        self.prior_pseudocount = s.get("prior_pseudocount", 0.1)

    @torch.no_grad()
    def sample(self, B: int, L: int):
        x0 = torch.distributions.Dirichlet(
            torch.ones(B, L, self.K, device=self.device)
        ).sample()
        xt = x0.clone()
        eye = torch.eye(self.K, device=self.device)

        t_span = torch.linspace(
            1, self.config.model.alpha_max, steps=self.n_steps, device=self.device
        )

        logits = None
        for s_t, t in zip(t_span[:-1], t_span[1:]):
            xt_expanded, _ = expand_simplex(
                xt, s_t[None].expand(B), self.prior_pseudocount
            )
            logits = self.model(xt_expanded, t=t[None].expand(B))
            flow_probs = F.softmax(logits / self.flow_temp, dim=-1)

            c_factor = self.condflow.c_factor(xt.detach().cpu().numpy(), s_t.item())
            c_factor = torch.from_numpy(c_factor).to(xt)

            cond_flows = (eye - xt.unsqueeze(-1)) * c_factor.unsqueeze(-2)
            flow = (flow_probs.unsqueeze(-2) * cond_flows).sum(-1)

            xt = (xt + flow * (t - s_t)).clamp(min=1e-8)
            xt = xt / xt.sum(-1, keepdim=True)

        seq_pred = torch.argmax(logits, dim=-1)
        return seq_pred, xt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--num_batches", type=int, default=20)
    args = parser.parse_args()

    with open(args.config) as f:
        config = ConfigDict(yaml.safe_load(f))
    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(config.get("seed", 42))

    sampler = UnconditionalSampler(config)
    B = config.loader.eval_batch_size
    L = config.dataset.seq_length

    out_path = os.path.join(args.out_dir, "unconditional_sequences.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seq"])
        for i in range(args.num_batches):
            seqs, _ = sampler.sample(B, L)
            for j in range(seqs.size(0)):
                writer.writerow([
                    "".join(INT_TO_BASE[t] for t in seqs[j].tolist()),
                ])
            print(f"[batch {i+1}/{args.num_batches}] {B} seqs written")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
