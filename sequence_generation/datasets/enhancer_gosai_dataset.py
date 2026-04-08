# Adapted from https://github.com/ChenyuWang-Monica/DRAKES/blob/master/drakes_dna/dataloader_gosai.py

import torch
import pandas as pd
import numpy as np
import os

DNA_ALPHABET = {'A': 0, 'C': 1, 'G': 2, 'T': 3} #, 'M': 4}

class GosaiDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        data_df = pd.read_csv(data_path)
        self.seqs = torch.tensor(data_df['seq'].apply(lambda x: [DNA_ALPHABET[c] for c in x]).tolist())
        self.clss = torch.tensor(data_df[['hepg2', 'k562', 'sknsh']].to_numpy())

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return {'seqs': self.seqs[idx], 'clss': self.clss[idx], 'attention_mask': torch.ones(len(self.seqs[idx]))}

def get_dataloaders_gosai(config, skip_valid=False, valid_seed=42):
    num_gpus = torch.cuda.device_count()
    if config.loader.global_batch_size % (num_gpus * config.train.accumulate_grad_batches) != 0:
        raise ValueError(
           f'Train Batch Size {config.train.batch_size}' f'not divisible by '
           f'{num_gpus} gpus with accumulation '
           f'{config.train.accumulate_grad_batches}.')
    if config.loader.eval_global_batch_size % num_gpus != 0:
        raise ValueError(
           f'Eval Batch Size for {config.eval.batch_size} '
           f'not divisible by {num_gpus}.')
    train_set = GosaiDataset(config.dataset.data_path)
    # randomly sample a subset of the train_set as valid_set
    valid_set = torch.utils.data.Subset(train_set, np.random.choice(len(train_set), 40000, replace=False))
    test_set = torch.utils.data.Subset(train_set, np.random.choice(len(train_set), 40000, replace=False))

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=config.loader.batch_size,
                                               num_workers=config.loader.num_workers,
                                               pin_memory=config.loader.pin_memory,
                                               shuffle=not config.dataset.streaming,persistent_workers=True)
    if skip_valid:
        valid_loader = None
        test_loader = None
    else:
        if valid_seed is None:
          shuffle_valid = False
          generator = None
        else:
          shuffle_valid = True
    generator = torch.Generator().manual_seed(valid_seed)
    valid_loader = torch.utils.data.DataLoader(valid_set,
                                               batch_size=config.loader.eval_batch_size,
                                               num_workers=config.loader.num_workers,
                                               pin_memory=config.loader.pin_memory,
                                               shuffle=shuffle_valid,generator=generator)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=config.loader.eval_batch_size,num_workers=config.loader.num_workers,
                                              pin_memory=config.loader.pin_memory,
                                              shuffle=shuffle_valid,generator=generator)
    return train_loader, valid_loader, test_loader
