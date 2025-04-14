import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import logging
import datetime
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, model, dataset, config):
        self.model = model
        self.dataset = dataset
        self.config = config
        
        # Setup device
        device_name = config.get('device', 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        
        if device_name == 'mps' and not torch.backends.mps.is_available():
            logging.warning("MPS device not available, falling back to CPU")
            device_name = 'cpu'
        
        if device_name == 'cuda' and not torch.cuda.is_available():
            logging.warning("CUDA device not available, falling back to CPU")
            device_name = 'cpu'
            
        self.device = torch.device(device_name)
        logging.info(f"Using device: {self.device}")
        
        self.model = self.model.to(self.device)
        
        # Setup data loader
        self.train_loader = DataLoader(
            dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['training']['num_workers']
        )
        
        # Setup optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Setup loss functions
        self.cls_criterion = nn.CrossEntropyLoss()
            
        self.checkpoint_dir = Path(config['training']['checkpoint_dir'])
        self.log_dir = Path(config['training']['log_dir'])
        
        self.seg_loss_weight = config['training']['seg_loss_weight']
        
        # Handle both save_interval (old) and save_frequency (new) parameter names
        if 'save_frequency' in config['training']:
            self.save_interval = config['training']['save_frequency']
        elif 'save_interval' in config['training']:
            self.save_interval = config['training']['save_interval']
        else:
            self.save_interval = 5  # Default value
            logging.warning("Neither save_frequency nor save_interval found in config, using default value of 5")
        
        logging.debug(f"Training configuration: {self.config}")
        
    def train(self):
        # Initialize history dictionary to track metrics
        history = {
            'loss': [],
            'cls_loss': [],
            'seg_loss': []
        }
        
        # Create log directory for plots
        plots_dir = self.log_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        num_epochs = self.config['training']['num_epochs']
        
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0
            epoch_cls_loss = 0
            epoch_seg_loss = 0
            
            logging.info(f'Epoch {epoch+1}/{num_epochs}')
            
            # Create progress bar for batches
            progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', 
                               leave=True, position=0)
            
            for batch_idx, batch in enumerate(progress_bar):
                # Handle different batch formats
                if isinstance(batch, dict):
                    images = batch['image'].to(self.device)
                    labels = batch['mask'].to(self.device)
                else:
                    # If batch is a tuple/list (common in PyTorch DataLoader)
                    images, labels = batch
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                
                # Add debug logging for the first batch
                if epoch == 0 and batch_idx == 0:
                    logging.debug(f"Input images shape: {images.shape}")
                    logging.debug(f"Input labels shape: {labels.shape}")
                    logging.debug(f"Input labels type: {labels.dtype}")
                
                # Forward pass with labels for weakly-supervised learning
                outputs = self.model(images, labels=labels) 
                logits = outputs['logits'] # Required for classification loss
                segmentation_maps = outputs['segmentation_maps'] # Required for segmentation loss
                
                # Calculate classification loss
                # Handle different label formats
                if labels.dim() > 1 and labels.size(1) > 1:
                    # If labels are one-hot encoded, convert to class indices
                    target_labels = torch.argmax(labels, dim=1)
                else:
                    # If labels are already class indices
                    target_labels = labels
                
                cls_loss = self.cls_criterion(logits, target_labels)
                
                # Calculate weak supervision loss
                seg_loss = self._calculate_weak_supervision_loss(segmentation_maps, labels)
                
                # Total loss
                loss = cls_loss + self.seg_loss_weight * seg_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Accumulate losses
                epoch_loss += loss.item()
                epoch_cls_loss += cls_loss.item()
                epoch_seg_loss += seg_loss.item()
                
                # Update progress bar with current loss
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'cls_loss': f'{cls_loss.item():.4f}',
                    'seg_loss': f'{seg_loss.item():.4f}'
                })
            
            # Calculate average epoch losses
            avg_epoch_loss = epoch_loss / len(self.train_loader)
            avg_cls_loss = epoch_cls_loss / len(self.train_loader)
            avg_seg_loss = epoch_seg_loss / len(self.train_loader)
            
            # Store losses in history
            history['loss'].append(avg_epoch_loss)
            history['cls_loss'].append(avg_cls_loss)
            history['seg_loss'].append(avg_seg_loss)
            logging.info(f'Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}, Cls Loss: {avg_cls_loss:.4f}, Seg Loss: {avg_seg_loss:.4f}')
            
            # Plot and save training progress
            if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
                self._plot_training_progress(history, self.log_dir / 'plots', epoch+1)
            
            # Save checkpoint
            if (epoch + 1) % self.save_interval == 0:
                self._save_checkpoint(epoch + 1, avg_epoch_loss, history)
                
    def _calculate_weak_supervision_loss(self, segmentation_maps, labels):
        """Calculate weak supervision loss for segmentation"""
        batch_size = segmentation_maps.size(0)
        num_classes = segmentation_maps.size(1)
        loss = 0
        
        # Convert label indices to one-hot encoding if they're not already
        if labels.dim() == 1 or (labels.dim() == 2 and labels.size(1) == 1):
            # If labels are class indices (shape: [batch_size] or [batch_size, 1])
            if labels.dim() == 2:
                labels = labels.squeeze(1)  # Convert [batch_size, 1] to [batch_size]
            
            # Create one-hot encoding
            one_hot_labels = torch.zeros(batch_size, num_classes, device=self.device)
            for i in range(batch_size):
                one_hot_labels[i, labels[i]] = 1
            labels = one_hot_labels
        
        # Now calculate Dice loss with properly formatted labels
        for i in range(batch_size):
            seg_map = segmentation_maps[i]
            label_idx = labels[i].argmax() if labels.dim() > 1 else labels[i]
            
            # Create a binary mask for the target class
            target_mask = torch.zeros_like(seg_map)
            target_mask[label_idx] = 1
            
            # Calculate Dice loss for the target class
            intersection = torch.sum(seg_map[label_idx] * target_mask[label_idx])
            union = torch.sum(seg_map[label_idx]) + torch.sum(target_mask[label_idx])
            dice_loss = 1 - (2 * intersection + 1e-6) / (union + 1e-6)
            loss += dice_loss
            
        return loss / batch_size
    
    def _plot_training_progress(self, history, plots_dir, epoch, final=False):
        """Plot and save training progress"""
        plt.figure(figsize=(12, 5))
        
        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(history['loss'], label='Total Loss')
        plt.plot(history['cls_loss'], label='Classification Loss')
        plt.plot(history['seg_loss'], label='Segmentation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training Losses')
        plt.grid(True)
        
        # Plot loss ratio
        plt.subplot(1, 2, 2)
        seg_loss_weight = self.seg_loss_weight
        weighted_seg_loss = [seg_loss_weight * s for s in history['seg_loss']]
        plt.stackplot(range(1, len(history['loss'])+1), 
                     [history['cls_loss'], weighted_seg_loss],
                     labels=['Classification Loss', 'Weighted Segmentation Loss'],
                     alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss Contribution')
        plt.legend(loc='upper right')
        plt.title('Loss Components')
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save the figure
        if final:
            plt.savefig(plots_dir / 'final_training_progress.png')
        else:
            plt.savefig(plots_dir / f'training_progress_epoch_{epoch}.png')
        
        plt.close()
    
    def _save_checkpoint(self, epoch, loss, history=None):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'config': self.config
        }
        
        if history is not None:
            checkpoint['history'] = history
        
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        logging.info(f'Saved checkpoint to {checkpoint_path}')
        
        # Save final model if this is the last epoch
        if epoch == self.config['training']['num_epochs']:
            final_path = self.checkpoint_dir / 'final_model.pth'
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'loss': loss,
                'history': history
            }, final_path)
            logging.info(f'Final model saved to {final_path}')
            
            # Final training plot
            self._plot_training_progress(history, self.log_dir / 'plots', epoch, final=True) 