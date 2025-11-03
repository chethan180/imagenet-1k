#\!/usr/bin/env python3
import wandb
import torch
import yaml

def test_wandb():
    print('Testing wandb integration...')
    
    # Load config
    with open('configs/imagenet1k_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize wandb
    wandb.init(
        project=config['logging']['project_name'],
        config=config,
        name='test-run',
        tags=['test', 'resnet50', 'imagenet1k']
    )
    
    # Log some test metrics
    for epoch in range(5):
        wandb.log({
            'epoch': epoch,
            'train_loss': 2.5 - epoch * 0.1,
            'train_accuracy': epoch * 10 + 50,
            'val_loss': 2.3 - epoch * 0.08,
            'val_accuracy': epoch * 8 + 60,
            'learning_rate': 0.1 * (0.9 ** epoch)
        })
    
    print('Wandb test completed successfully\!')
    wandb.finish()

if __name__ == '__main__':
    test_wandb()
