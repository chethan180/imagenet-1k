import os
import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
import random

from src.models import get_model
from src.data import ImageNetDataset
from src.utils import get_criterion, get_optimizer, get_scheduler_with_warmup, Logger
from src.training import Trainer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='ImageNet Training')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='path to checkpoint to resume from')
    parser.add_argument('--data-path', type=str, default=None,
                        help='path to dataset (overrides config)')
    parser.add_argument('--dataset', type=str, choices=['imagenet1k', 'imagenet100k'], default=None,
                        help='dataset type (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='batch size (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                        help='learning rate (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='number of epochs (overrides config)')
    parser.add_argument('--model', type=str, default=None,
                        help='model name (overrides config)')
    parser.add_argument('--no-wandb', action='store_true',
                        help='disable wandb logging')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    if args.data_path:
        config['data']['data_path'] = args.data_path
    if args.dataset:
        config['data']['dataset'] = args.dataset
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['lr'] = args.lr
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.model:
        config['model']['name'] = args.model
    if args.no_wandb:
        config['logging']['use_wandb'] = False
    
    set_seed(config['misc']['seed'])
    
    device = torch.device(config['misc']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if config['data']['dataset'] == 'imagenet100k':
        num_classes = 1000
    else:
        num_classes = config['model']['num_classes']
    
    model = get_model(
        config['model']['name'],
        num_classes=num_classes,
        drop_path_rate=0.1 if 'resnet' in config['model']['name'] else 0.0
    )
    model = model.to(device)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs")
    
    dataset = ImageNetDataset(config)
    train_loader, val_loader = dataset.get_dataloaders()
    
    criterion = get_criterion(config)
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler_with_warmup(optimizer, config)
    
    logger = Logger(config)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        logger=logger,
        device=device
    )
    
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        logger.close()


if __name__ == '__main__':
    main()