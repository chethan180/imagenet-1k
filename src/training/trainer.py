import torch
import torch.nn as nn
import time
from tqdm import tqdm
import numpy as np
import random

from ..data import mixup_data, cutmix_data
from ..utils import AverageMeter, AccuracyMeter, MixupLoss


class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler, 
                 config, logger, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.logger = logger
        self.device = device
        
        self.mixup_criterion = MixupLoss(self.criterion)
        self.scaler = torch.cuda.amp.GradScaler() if config['misc']['amp'] else None
        
        self.best_acc = 0.0
        self.start_epoch = 0
        
        if config['misc']['compile'] and hasattr(torch, 'compile'):
            self.model = torch.compile(self.model)
            self.logger.logger.info("Model compiled with torch.compile")

    def train_epoch(self, epoch):
        self.model.train()
        
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        acc_meter = AccuracyMeter()
        
        end = time.time()
        
        pbar = tqdm(self.train_loader, desc=f'Train Epoch {epoch:03d}')
        
        for i, (images, targets) in enumerate(pbar):
            data_time.update(time.time() - end)
            
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            r = np.random.rand(1)
            
            if self.config['training'].get('mixup_alpha', 0) > 0 and r < 0.5:
                images, targets_a, targets_b, lam = mixup_data(
                    images, targets, self.config['training']['mixup_alpha']
                )
                use_mixup = True
            elif self.config['training'].get('cutmix_alpha', 0) > 0 and r < 0.8:
                images, targets_a, targets_b, lam = cutmix_data(
                    images, targets, self.config['training']['cutmix_alpha']
                )
                use_mixup = True
            else:
                use_mixup = False
            
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    if use_mixup:
                        loss = self.mixup_criterion(outputs, targets_a, targets_b, lam)
                    else:
                        loss = self.criterion(outputs, targets)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                if use_mixup:
                    loss = self.mixup_criterion(outputs, targets_a, targets_b, lam)
                else:
                    loss = self.criterion(outputs, targets)
                
                loss.backward()
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            
            losses.update(loss.item(), images.size(0))
            if not use_mixup:
                acc_meter.update(outputs, targets, images.size(0))
            
            batch_time.update(time.time() - end)
            end = time.time()
            
            if i % self.config['logging']['log_interval'] == 0:
                lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'Loss': f'{losses.avg:.4f}',
                    'Top1': f'{acc_meter.get_avg()["top1"]:.2f}%' if not use_mixup else 'N/A',
                    'LR': f'{lr:.6f}'
                })
                
                self.logger.log_metrics({
                    'train_loss_batch': losses.val,
                    'train_lr': lr,
                    'batch_time': batch_time.avg,
                    'data_time': data_time.avg
                }, step=epoch * len(self.train_loader) + i)
        
        metrics = {
            'loss': losses.avg,
            **acc_meter.get_avg()
        }
        
        return metrics

    def validate(self, epoch):
        self.model.eval()
        
        batch_time = AverageMeter()
        losses = AverageMeter()
        acc_meter = AccuracyMeter()
        
        with torch.no_grad():
            end = time.time()
            pbar = tqdm(self.val_loader, desc=f'Val Epoch {epoch:03d}')
            
            for i, (images, targets) in enumerate(pbar):
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
                
                losses.update(loss.item(), images.size(0))
                acc_meter.update(outputs, targets, images.size(0))
                
                batch_time.update(time.time() - end)
                end = time.time()
                
                pbar.set_postfix({
                    'Loss': f'{losses.avg:.4f}',
                    'Top1': f'{acc_meter.get_avg()["top1"]:.2f}%',
                    'Top5': f'{acc_meter.get_avg()["top5"]:.2f}%'
                })
        
        metrics = {
            'loss': losses.avg,
            **acc_meter.get_avg()
        }
        
        return metrics

    def train(self):
        self.logger.log_model_info(self.model)
        
        for epoch in range(self.start_epoch, self.config['training']['epochs']):
            
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)
            
            if hasattr(self.scheduler, 'step'):
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            lr = self.optimizer.param_groups[0]['lr']
            self.logger.log_epoch_summary(epoch, train_metrics, val_metrics, lr)
            
            is_best = val_metrics['top1'] > self.best_acc
            if is_best:
                self.best_acc = val_metrics['top1']
            
            if (epoch + 1) % self.config['logging']['save_interval'] == 0 or is_best:
                state = {
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict() if self.scheduler else None,
                    'best_acc': self.best_acc,
                    'config': self.config
                }
                self.logger.save_checkpoint(state, epoch + 1, is_best)
        
        self.logger.logger.info(f"Training completed. Best validation accuracy: {self.best_acc:.3f}%")
        
    def load_checkpoint(self, checkpoint_path):
        self.logger.logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.start_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.scheduler and checkpoint['scheduler']:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.best_acc = checkpoint['best_acc']
        
        self.logger.logger.info(f"Loaded checkpoint (epoch {self.start_epoch}, best_acc {self.best_acc:.3f}%)")