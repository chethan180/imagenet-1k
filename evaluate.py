import os
import argparse
import yaml
import torch
import torch.nn as nn
from tqdm import tqdm

from src.models import get_model
from src.data import ImageNetDataset
from src.utils import AccuracyMeter, AverageMeter


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def evaluate_model(model, val_loader, device, amp=False):
    model.eval()
    
    acc_meter = AccuracyMeter()
    losses = AverageMeter()
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Evaluating')
        
        for images, targets in pbar:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            if amp:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, targets)
            else:
                outputs = model(images)
                loss = criterion(outputs, targets)
            
            losses.update(loss.item(), images.size(0))
            acc_meter.update(outputs, targets, images.size(0))
            
            pbar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'Top1': f'{acc_meter.get_avg()["top1"]:.2f}%',
                'Top5': f'{acc_meter.get_avg()["top5"]:.2f}%'
            })
    
    return {
        'loss': losses.avg,
        **acc_meter.get_avg()
    }


def main():
    parser = argparse.ArgumentParser(description='ImageNet Evaluation')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='path to checkpoint')
    parser.add_argument('--data-path', type=str, default=None,
                        help='path to dataset (overrides config)')
    parser.add_argument('--dataset', type=str, choices=['imagenet1k', 'imagenet100k'], default=None,
                        help='dataset type (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='batch size (overrides config)')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    if args.data_path:
        config['data']['data_path'] = args.data_path
    if args.dataset:
        config['data']['dataset'] = args.dataset
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if config['data']['dataset'] == 'imagenet100k':
        num_classes = 1000
    else:
        num_classes = config['model']['num_classes']
    
    model = get_model(config['model']['name'], num_classes=num_classes)
    
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    if list(state_dict.keys())[0].startswith('module.'):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        state_dict = new_state_dict
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs")
    
    dataset = ImageNetDataset(config)
    _, val_loader = dataset.get_dataloaders()
    
    print("Starting evaluation...")
    results = evaluate_model(model, val_loader, device, amp=config['misc'].get('amp', False))
    
    print("\nEvaluation Results:")
    print(f"Loss: {results['loss']:.4f}")
    print(f"Top-1 Accuracy: {results['top1']:.2f}%")
    print(f"Top-5 Accuracy: {results['top5']:.2f}%")
    
    if 'best_acc' in checkpoint:
        print(f"Checkpoint Best Accuracy: {checkpoint['best_acc']:.2f}%")


if __name__ == '__main__':
    main()