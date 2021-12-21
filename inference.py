import torch
import torch.nn as nn

from options import args
from models import model_factory
from dataloaders import dataloader_factory
from trainers import trainer_factory
from utils import *
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2' # specify GPUs locally
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def inference():
    export_root = setup_train(args)
    train_loader, val_loader, test_loader = dataloader_factory(args)
    model = model_factory(args)

    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, export_root)
    trainer.test()


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    else:
        raise ValueError('Invalid mode')
