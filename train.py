import torch
import torch.nn as nn

from options import args
from models import model_factory
from dataloaders import dataloader_factory
from trainers import trainer_factory
from utils import *
import os
import pandas as pd

# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3' # specify GPUs locally
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # specify GPUs locally
torch.cuda.set_device(int(args.device_idx))
device = torch.device(f'cuda:{args.device_idx}' if torch.cuda.is_available() else 'cpu')

def train():
    export_root = setup_train(args)
    train_loader, val_loader, test_loader = dataloader_factory(args)
    model = model_factory(args)
    model = model.to(device)
    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, export_root)
    trainer.train()
    trainer.test()


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    else:
        raise ValueError('Invalid mode')
