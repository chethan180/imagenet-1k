import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, StepLR
import math


def get_optimizer(model, config):
    optimizer_config = config['optimizer']
    training_config = config['training']
    
    if optimizer_config['name'].lower() == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=float(training_config['lr']),
            momentum=float(optimizer_config.get('momentum', 0.9)),
            weight_decay=float(optimizer_config.get('weight_decay', 1e-4)),
            nesterov=optimizer_config.get('nesterov', True)
        )
    elif optimizer_config['name'].lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=float(training_config['lr']),
            weight_decay=float(optimizer_config.get('weight_decay', 1e-4)),
            betas=optimizer_config.get('betas', (0.9, 0.999))
        )
    elif optimizer_config['name'].lower() == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=float(training_config['lr']),
            weight_decay=float(optimizer_config.get('weight_decay', 1e-4)),
            betas=optimizer_config.get('betas', (0.9, 0.999))
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_config['name']}")
    
    return optimizer


def get_scheduler(optimizer, config):
    scheduler_config = config['scheduler']
    training_config = config['training']
    
    if scheduler_config['name'].lower() == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=training_config['epochs'] - scheduler_config.get('warmup_epochs', 0),
            eta_min=scheduler_config.get('min_lr', 1e-5)
        )
    elif scheduler_config['name'].lower() == 'step':
        scheduler = StepLR(
            optimizer,
            step_size=scheduler_config.get('step_size', 30),
            gamma=scheduler_config.get('gamma', 0.1)
        )
    elif scheduler_config['name'].lower() == 'multistep':
        scheduler = MultiStepLR(
            optimizer,
            milestones=scheduler_config.get('milestones', [30, 60, 80]),
            gamma=scheduler_config.get('gamma', 0.1)
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_config['name']}")
    
    return scheduler


class WarmupScheduler:
    def __init__(self, optimizer, scheduler, warmup_epochs, warmup_lr, base_lr):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.warmup_epochs = warmup_epochs
        self.warmup_lr = warmup_lr
        self.base_lr = base_lr
        self.current_epoch = 0
        
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.current_epoch + 1
        self.current_epoch = epoch
        
        if epoch <= self.warmup_epochs:
            lr = self.warmup_lr + (self.base_lr - self.warmup_lr) * epoch / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            self.scheduler.step(epoch - self.warmup_epochs)


def get_scheduler_with_warmup(optimizer, config):
    scheduler = get_scheduler(optimizer, config)
    scheduler_config = config['scheduler']
    training_config = config['training']
    
    warmup_epochs = scheduler_config.get('warmup_epochs', 0)
    if warmup_epochs > 0:
        return WarmupScheduler(
            optimizer=optimizer,
            scheduler=scheduler,
            warmup_epochs=warmup_epochs,
            warmup_lr=scheduler_config.get('warmup_lr', 1e-6),
            base_lr=training_config['lr']
        )
    else:
        return scheduler