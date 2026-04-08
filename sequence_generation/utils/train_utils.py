from pathlib import Path
import torch
import random
import numpy as np
import os

from sequence_generation.datasets.enhancer_gosai_dataset import get_dataloaders_gosai
from sequence_generation.model import CNNModel, SeqRegressor

def load_seed(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms = True


def load_generator(config):
    if config.name == "promoter_model":
        # TODO: check type of model
        model = CNNModel(config.model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.lr)
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.train.lr_step_size, gamma=config.train.lr_gamma)
    if config.name == "enhancer_model":
        model = CNNModel(config.model, alphabet_size=config.model.alphabet_size, num_cls=config.model.num_cls)
        if config.model.checkpoint_path:
            checkpoint = torch.load(config.model.checkpoint_path, map_location='cpu')
            # TODO: Since checkpoint is saved as lightning module, need to remove "model." prefix at the moment
            if any(k.startswith("model.") for k in checkpoint['state_dict'].keys()):
                checkpoint['state_dict'] = {k[len("model."):]: v for k, v in checkpoint['state_dict'].items()}
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.lr)
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.train.lr_step_size, gamma=config.train.lr_gamma)
        
    lr_scheduler = None
    return model, optimizer, lr_scheduler

def load_regressor(config):
    model = SeqRegressor(config.model.alphabet_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.lr) 
    lr_scheduler = None

    return model, optimizer, lr_scheduler

def load_dataloader(config):
    if config.dataset.name == 'gosai':
        train_loader, val_loader, test_loader = get_dataloaders_gosai(config)
    return train_loader, val_loader, test_loader

def save_checkpoint(
    model, optimizer, lr_scheduler, output_dir, global_step, epoch_index, logger=None):
    save_path = Path(output_dir) / f"checkpoint-epoch{epoch_index}.pth"
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        "global_step": global_step,
    }
    torch.save(checkpoint, save_path)
    if logger:
        logger.info(f"Checkpoint saved at {save_path}")
    else:
        print(f"Checkpoint saved at {save_path}")
    return save_path