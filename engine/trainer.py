import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

from utils.losses import get_loss_function
from utils.metrics import SegmentationMetrics
from datasets import create_data_loaders

class EarlyStopping:
    """Early stopping mechanism."""
    
    def __init__(self, patience=7, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if mode == 'max':
            self.compare = lambda score, best: score > best + self.min_delta
        else:
            self.compare = lambda score, best: score < best - self.min_delta
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.compare(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop

class Trainer:
    """Model trainer."""
    
    def __init__(self, model, config, device=None):
        self.model = model
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Create data loaders
        self.train_loader, self.val_loader, self.test_loader, self.pos_weight = create_data_loaders(config)
        
        # Loss function
        self.criterion = get_loss_function(config, self.pos_weight)
        
        # Optimizer & LR scheduler
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()
        
        # Metric trackers
        self.train_metrics = SegmentationMetrics()
        self.val_metrics = SegmentationMetrics()
        
        # Early stopping
        early_stop_config = config.get('early_stopping', {})
        self.early_stopping = EarlyStopping(
            patience=early_stop_config.get('patience', 10),
            min_delta=early_stop_config.get('min_delta', 0.001),
            mode=early_stop_config.get('mode', 'max')
        )
        
        # Logging
        self.writer = None
        self.log_dir = config.get('logging', {}).get('log_dir', 'outputs/logs')
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_dice': [],
            'val_loss': [],
            'val_dice': [],
            'learning_rates': []
        }
        
        # Checkpoint directory
        self.save_dir = config.get('logging', {}).get('save_dir', 'checkpoints')
        os.makedirs(self.save_dir, exist_ok=True)
        
        print(f"Training on device: {self.device}")
        print(f"Dataset sizes - Train: {len(self.train_loader.dataset)}, "
              f"Val: {len(self.val_loader.dataset)}, Test: {len(self.test_loader.dataset)}")
        print(f"Positive weight: {self.pos_weight:.3f}")
    
    def _get_optimizer(self):
        """Get optimizer."""
        optimizer_name = self.config['training'].get('optimizer', 'adam').lower()
        lr = self.config['training']['learning_rate']
        weight_decay = self.config['training'].get('weight_decay', 1e-4)
        
        if optimizer_name == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            return torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        elif optimizer_name == 'adamw':
            return torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _get_scheduler(self):
       """Get learning rate scheduler."""
       scheduler_config = self.config.get('scheduler', {})
       if not scheduler_config:
           return None

       scheduler_type = scheduler_config.get('type', 'reduce_on_plateau')

       if scheduler_type == 'reduce_on_plateau':
           # More explicit parameters for stable behavior
           factor    = float(scheduler_config.get('factor', 0.5))
           patience  = int(scheduler_config.get('patience', 4))
           min_lr    = float(scheduler_config.get('min_lr', 1e-5))
           threshold = float(scheduler_config.get('threshold', 1e-3))

           return torch.optim.lr_scheduler.ReduceLROnPlateau(
               self.optimizer,
               mode='max',             # maximize Dice
               factor=factor,
               patience=patience,
               min_lr=min_lr,
               threshold=threshold,    # minimal improvement to be considered as progress
               verbose=True
           )

       elif scheduler_type == 'step':
           return torch.optim.lr_scheduler.StepLR(
               self.optimizer,
               step_size=int(scheduler_config.get('step_size', 20)),
               gamma=float(scheduler_config.get('gamma', 0.1)),
           )

       elif scheduler_type == 'cosine':
           return torch.optim.lr_scheduler.CosineAnnealingLR(
               self.optimizer,
               T_max=int(self.config['training']['num_epochs'])
           )
       else:
           return None

    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        self.train_metrics.reset()
        
        running_loss = 0.0
        num_batches = len(self.train_loader)
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc='Training', leave=False)
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            running_loss += loss.item()
            self.train_metrics.update(outputs, masks)
            
            # Update progress bar
            current_loss = running_loss / (batch_idx + 1)
            pbar.set_postfix({'Loss': f'{current_loss:.4f}'})
        
        # Epoch metrics
        epoch_loss = running_loss / num_batches
        epoch_metrics = self.train_metrics.compute()
        
        return epoch_loss, epoch_metrics
    
    def validate_epoch(self):
        """Validate for one epoch."""
        self.model.eval()
        self.val_metrics.reset()
        
        running_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation', leave=False)
            
            for batch_idx, batch in enumerate(pbar):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # Forward
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                # Update metrics
                running_loss += loss.item()
                self.val_metrics.update(outputs, masks)
                
                # Update progress bar
                current_loss = running_loss / (batch_idx + 1)
                pbar.set_postfix({'Loss': f'{current_loss:.4f}'})
        
        # Epoch metrics
        epoch_loss = running_loss / num_batches
        epoch_metrics = self.val_metrics.compute()
        
        return epoch_loss, epoch_metrics
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.save_dir, 'latest_checkpoint.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"Best model saved to {best_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint['epoch'], checkpoint['metrics']
    
    def train(self, num_epochs=None, resume_from=None):
        """Train the model."""
        if num_epochs is None:
            num_epochs = self.config['training']['num_epochs']
        
        start_epoch = 0
        best_dice = 0.0
        
        # Resume training
        if resume_from:
            start_epoch, metrics = self.load_checkpoint(resume_from)
            best_dice = metrics.get('dice', 0.0)
            print(f"Resumed training from epoch {start_epoch}")
        
        # TensorBoard
        if self.writer is None:
            self.writer = SummaryWriter(self.log_dir)
        
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(start_epoch, num_epochs):
            epoch_start_time = time.time()
            
            # Train
            train_loss, train_metrics = self.train_epoch()
            
            # Validate
            val_loss, val_metrics = self.validate_epoch()
            
            # LR scheduling
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['dice'])
                else:
                    self.scheduler.step()
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_dice'].append(train_metrics['dice'])
            self.history['val_loss'].append(val_loss)
            self.history['val_dice'].append(val_metrics['dice'])
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # TensorBoard logs
            if self.writer:
                self.writer.add_scalars('Loss', {
                    'Train': train_loss,
                    'Validation': val_loss
                }, epoch)
                
                self.writer.add_scalars('Dice Score', {
                    'Train': train_metrics['dice'],
                    'Validation': val_metrics['dice']
                }, epoch)
                
                self.writer.add_scalar('Learning Rate', 
                                     self.optimizer.param_groups[0]['lr'], epoch)
            
            # Best model check
            is_best = val_metrics['dice'] > best_dice
            if is_best:
                best_dice = val_metrics['dice']
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Early stopping
            if self.early_stopping(val_metrics['dice']):
                print(f"Early stopping triggered at epoch {epoch}")
                break
            
            # Console logging
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch+1}/{num_epochs} - {epoch_time:.2f}s")
            print(f"  Train - Loss: {train_loss:.4f}, Dice: {train_metrics['dice']:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, Dice: {val_metrics['dice']:.4f}, IoU: {val_metrics['iou']:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Print detailed metrics every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"  Detailed Val Metrics:")
                print(f"    Precision: {val_metrics['precision']:.4f}")
                print(f"    Recall: {val_metrics['recall']:.4f}")
                print(f"    Accuracy: {val_metrics['accuracy']:.4f}")
        
        if self.writer:
            self.writer.close()
        
        print(f"Training completed! Best validation Dice: {best_dice:.4f}")
        return self.history
    
    def plot_training_curves(self, save_path=None):
        """Plot training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss', alpha=0.8)
        axes[0, 0].plot(self.history['val_loss'], label='Validation Loss', alpha=0.8)
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Dice curves
        axes[0, 1].plot(self.history['train_dice'], label='Train Dice', alpha=0.8)
        axes[0, 1].plot(self.history['val_dice'], label='Validation Dice', alpha=0.8)
        axes[0, 1].set_title('Training and Validation Dice Score')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Dice Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate
        axes[1, 0].plot(self.history['learning_rates'], alpha=0.8)
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
        
        # Combined validation view
        axes[1, 1].plot(self.history['val_loss'], label='Val Loss (scaled)', alpha=0.8)
        axes[1, 1].plot(self.history['val_dice'], label='Val Dice', alpha=0.8)
        axes[1, 1].set_title('Validation Metrics Comparison')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

# Convenience training function
def train_model_from_config(config_path, model_class, resume_from=None):
    """Train a model from a YAML config."""
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Build model
    if hasattr(model_class, 'from_config'):
        model = model_class.from_config(config)
    else:
        model = model_class(**config['model'])
    
    # Build trainer
    trainer = Trainer(model, config)
    
    # Train
    history = trainer.train(resume_from=resume_from)
    
    return trainer, history

