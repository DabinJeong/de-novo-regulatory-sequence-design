from scripts.masked_separator_trainer import MaskedSeparatorTrainer 
from scripts.sampler import Sampler

import argparse
import yaml
from ml_collections.config_dict import ConfigDict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for sequence generation model")
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory for logs and checkpoints')
    # mutually exclusive group: arguments should be either --train or --generate
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--train', action='store_true', help='Flag to indicate training mode')
    mode_group.add_argument('--generate', action='store_true', help='Flag to indicate sampling mode')
    args = parser.parse_args()

    # Load configuration

    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    config = ConfigDict(config_dict)

    if args.train:
        # Initialize Trainer
        trainer = MaskedSeparatorTrainer(config)

        # Start training
        trainer.train()
    elif args.generate:
        # Initialize Sampler
        sampler = Sampler(config)
        B, L = config.train.batch_size, config.dataset.seq_length
        # Start sampling
        sampler.sample(B, L)
    