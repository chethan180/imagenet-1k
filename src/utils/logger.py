import os
import logging
import wandb
from datetime import datetime
import torch
import json
import matplotlib.pyplot as plt


class Logger:
    def __init__(self, config, run_name=None):
        self.config = config
        self.use_wandb = config['logging']['use_wandb']
        self.log_interval = config['logging']['log_interval']
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = run_name or f"{config['model']['name']}_{timestamp}"
        
        self.train_history = {'loss': [], 'f1': [], 'precision': [], 'recall': []}
        self.val_history = {'loss': [], 'f1': [], 'precision': [], 'recall': []}
        self.epochs = []

        self.setup_local_logging()
        if self.use_wandb:
            self.setup_wandb()
    
    def setup_local_logging(self):
        base_log_dir = self.config['logging'].get('log_dir', os.path.join(os.getcwd(), 'logs'))
        log_dir = os.path.join(base_log_dir, self.run_name)
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, 'training.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.log_dir = log_dir
        
        config_file = os.path.join(log_dir, 'config.yaml')
        with open(config_file, 'w') as f:
            import yaml
            yaml.dump(self.config, f, default_flow_style=False)
    
    def setup_wandb(self):
        wandb.init(
            project=self.config['logging']['project_name'],
            name=self.run_name,
            config=self.config,
            reinit=True
        )
        
    def log_metrics(self, metrics, step=None, prefix=""):
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        
        self.logger.info(f"Step {step}: {metrics}")
        
        if self.use_wandb:
            wandb.log(metrics, step=step)
    
    def log_model_info(self, model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        info = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }
        
        self.logger.info(f"Model info: {info}")
        
        if self.use_wandb:
            wandb.log(info)
    
    def save_checkpoint(self, state, epoch, is_best=False):
        base_checkpoint_dir = self.config['logging'].get('checkpoint_dir', os.path.join(self.log_dir, 'checkpoints'))
        checkpoint_dir = os.path.join(base_checkpoint_dir, self.run_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        filename = f'checkpoint_epoch_{epoch:03d}.pth'
        filepath = os.path.join(checkpoint_dir, filename)
        
        torch.save(state, filepath)
        
        if is_best:
            best_filepath = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save(state, best_filepath)
            self.logger.info(f"New best model saved with validation accuracy: {state['best_acc']:.3f}%")
        
        self.logger.info(f"Checkpoint saved: {filepath}")
    
    def log_epoch_summary(self, epoch, train_metrics, val_metrics, lr):
        summary = {
            'epoch': epoch,
            'learning_rate': lr,
            **{f'train_{k}': v for k, v in train_metrics.items()},
            **{f'val_{k}': v for k, v in val_metrics.items()}
        }
        
        # --- Save metrics history for plotting ---
        self.epochs.append(epoch)
        for key in ['loss', 'f1', 'precision', 'recall']:
            self.train_history[key].append(train_metrics.get(key, float('nan')))
            self.val_history[key].append(val_metrics.get(key, float('nan')))
        
        self.log_metrics(summary, step=epoch)
        
        # Create comprehensive log message
        log_msg = (
            f"Epoch {epoch:03d} Summary - "
            f"LR: {lr:.6f}, "
            f"Train Loss: {train_metrics.get('loss', 0):.4f}, "
            f"Val Loss: {val_metrics.get('loss', 0):.4f}"
        )
        
        # Add additional metrics
        for m in ['f1', 'precision', 'recall']:
            if m in val_metrics:
                log_msg += f", Val {m.capitalize()}: {val_metrics[m]:.2f}%"
        
        self.logger.info(log_msg)
        
        # --- Generate charts ---
        self._plot_metrics()

    def _plot_metrics(self):
        """Generate and save charts for train/val metrics."""
        plot_dir = os.path.join(self.log_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)

        metrics_to_plot = ['loss', 'f1', 'precision', 'recall']
        plt.figure(figsize=(12, 8))

        for i, m in enumerate(metrics_to_plot, 1):
            plt.subplot(2, 2, i)
            plt.plot(self.epochs, self.train_history[m], label=f"Train {m}")
            plt.plot(self.epochs, self.val_history[m], label=f"Val {m}")
            plt.title(m.upper())
            plt.xlabel("Epoch")
            plt.ylabel(m)
            plt.legend()
            plt.grid(True)

        plt.tight_layout()
        save_path = os.path.join(plot_dir, "metrics_epochwise.png")
        plt.savefig(save_path)
        plt.close()

        if self.use_wandb:
            wandb.log({"metrics_chart": wandb.Image(save_path)})

    def close(self):
        if self.use_wandb:
            wandb.finish()
        
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)
