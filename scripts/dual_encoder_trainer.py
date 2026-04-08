"""
Trainer for the dual (semantic / variation) encoder.

Usage:
    python -m scripts.dual_encoder_trainer \
        --config configs/enhancer_gosai_dual_encoder.yaml \
        --out_dir ./runs/dual_encoder
"""

import os
import argparse
import yaml
import torch
from torch.optim import AdamW
from tqdm import tqdm
from ml_collections.config_dict import ConfigDict

from sequence_generation.utils.train_utils import load_dataloader, load_seed
from sequence_generation.model.dual_encoder import DualEncoderModel


def env_id_from_batch(batch, num_envs: int) -> torch.Tensor:
    """
    Gosai batches expose 'clss' = (B, num_tasks). We pick the env label as
    argmax of activity across tasks (i.e. the cell line where the sequence
    is most active). Override here for a different env definition.
    """
    return batch["clss"].argmax(dim=-1).clamp(max=num_envs - 1)


class DualEncoderTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        load_seed(config.seed)
        self.train_loader, self.eval_loader, _ = load_dataloader(config)

        self.num_envs = config.get("dual_encoder", {}).get("num_envs", 3)
        self.model = DualEncoderModel(config, num_envs=self.num_envs).to(self.device)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.train.lr,
            weight_decay=config.optim.get("weight_decay", 0.0),
        )

        self.out_dir = config.get("out_dir", "./runs/dual_encoder")
        os.makedirs(self.out_dir, exist_ok=True)

    def train(self):
        best = float("inf")
        for epoch in range(self.config.train.num_epochs):
            self.model.train()
            agg = {"loss": 0.0, "recon": 0.0, "inv": 0.0, "div": 0.0}
            n = 0
            for batch in tqdm(self.train_loader, desc=f"epoch {epoch+1}"):
                x       = batch["seqs"].to(self.device)
                env_ids = env_id_from_batch(batch, self.num_envs).to(self.device)

                self.optimizer.zero_grad()
                losses = self.model.compute_loss(x, env_ids)
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                for k in agg:
                    agg[k] += losses[k].item()
                n += 1

            avg = {k: v / max(n, 1) for k, v in agg.items()}
            print(f"epoch {epoch+1:3d}  loss={avg['loss']:.4f}  "
                  f"recon={avg['recon']:.4f}  inv={avg['inv']:.4f}  div={avg['div']:.4f}")

            if avg["loss"] < best:
                best = avg["loss"]
                ckpt = os.path.join(self.out_dir, "dual_encoder_best.ckpt")
                torch.save({"epoch": epoch, "model": self.model.state_dict(),
                            "loss": best}, ckpt)
                print(f"  -> saved {ckpt}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = ConfigDict(yaml.safe_load(f))
    config.out_dir = args.out_dir

    DualEncoderTrainer(config).train()


if __name__ == "__main__":
    main()
