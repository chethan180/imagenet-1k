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
            eta_min=float(scheduler_config.get('min_lr', 1e-5))  # ✅ ensure float
        )
    elif scheduler_config['name'].lower() == 'step':
        scheduler = StepLR(
            optimizer,
            step_size=int(scheduler_config.get('step_size', 30)),
            gamma=float(scheduler_config.get('gamma', 0.1))
        )
    elif scheduler_config['name'].lower() == 'multistep':
        scheduler = MultiStepLR(
            optimizer,
            milestones=list(map(int, scheduler_config.get('milestones', [30, 60, 80]))),
            gamma=float(scheduler_config.get('gamma', 0.1))
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_config['name']}")
    
    return scheduler



class WarmupScheduler:
    def __init__(self, optimizer, scheduler, warmup_epochs, warmup_lr, base_lr):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.warmup_epochs = int(warmup_epochs)
        self.warmup_lr = float(warmup_lr)
        self.base_lr = float(base_lr)
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
    
    def state_dict(self):
        return {
            'current_epoch': self.current_epoch,
            'warmup_epochs': self.warmup_epochs,
            'warmup_lr': self.warmup_lr,
            'base_lr': self.base_lr,
            'scheduler_state_dict': self.scheduler.state_dict()
        }
    
    def load_state_dict(self, state_dict):
        self.current_epoch = state_dict['current_epoch']
        self.warmup_epochs = state_dict['warmup_epochs']
        self.warmup_lr = state_dict['warmup_lr']
        self.base_lr = state_dict['base_lr']
        self.scheduler.load_state_dict(state_dict['scheduler_state_dict'])


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
            warmup_lr=float(scheduler_config.get('warmup_lr', 1e-6)),
            base_lr=float(training_config['lr'])
        )
    else:
        return scheduler