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
                if labels.dim() > 1:
                    if labels.size(1) > 1 and labels.size(1) <= self.config['model']['num_classes']:
                        # If labels are one-hot encoded, convert to class indices
                        target_labels = torch.argmax(labels, dim=1)
                    else:
                        # If labels are segmentation masks (H,W) format, extract class indices
                        # For segmentation masks, we'll use the most common class in the mask
                        # First, flatten the spatial dimensions
                        if labels.dim() == 3:  # [batch_size, H, W]
                            flat_labels = labels.reshape(labels.size(0), -1)
                        elif labels.dim() == 4:  # [batch_size, C, H, W]
                            # If it's a one-hot segmentation mask
                            flat_labels = labels.argmax(dim=1).reshape(labels.size(0), -1)
                        else:
                            # Unexpected format, use zeros as fallback
                            target_labels = torch.zeros(labels.size(0), dtype=torch.long, device=self.device)
                            print(f"Warning: Unexpected label format: {labels.shape}")
                            
                        # For each sample, find the most common class (mode)
                        target_labels = torch.zeros(labels.size(0), dtype=torch.long, device=self.device)
                        for i in range(labels.size(0)):
                            # Count occurrences of each class
                            unique_values, counts = torch.unique(flat_labels[i], return_counts=True)
                            # Find the most common class (excluding background class 0 if possible)
                            if len(unique_values) > 1 and 0 in unique_values:
                                # Filter out background class
                                mask = unique_values != 0
                                filtered_values = unique_values[mask]
                                filtered_counts = counts[mask]
                                if len(filtered_values) > 0:
                                    # Use the most common non-background class
                                    target_labels[i] = filtered_values[filtered_counts.argmax()]
                                else:
                                    # If only background remains, use it
                                    target_labels[i] = 0
                            else:
                                # Use the most common class
                                target_labels[i] = unique_values[counts.argmax()]
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
        
        # Handle different label formats
        if labels.dim() == 1:  # Simple class indices [batch_size]
            # Create one-hot encoding
            one_hot_labels = torch.zeros(batch_size, num_classes, device=self.device)
            for i in range(batch_size):
                one_hot_labels[i, labels[i]] = 1
            class_labels = one_hot_labels
            
        elif labels.dim() == 2:
            if labels.size(1) == 1:  # Class indices as [batch_size, 1]
                # Create one-hot encoding
                one_hot_labels = torch.zeros(batch_size, num_classes, device=self.device)
                for i in range(batch_size):
                    one_hot_labels[i, labels[i, 0]] = 1
                class_labels = one_hot_labels
                
            elif labels.size(1) == num_classes:  # Already one-hot [batch_size, num_classes]
                class_labels = labels
                
            else:  # Unexpected format, try to handle it
                print(f"Warning: Unexpected label format in weak supervision loss: {labels.shape}")
                # Default to first class
                class_labels = torch.zeros(batch_size, num_classes, device=self.device)
                class_labels[:, 0] = 1
                
        elif labels.dim() == 3:  # Segmentation masks [batch_size, H, W]
            # Extract class labels from segmentation masks
            class_labels = torch.zeros(batch_size, num_classes, device=self.device)
            for i in range(batch_size):
                # Count occurrences of each class
                flat_mask = labels[i].reshape(-1)
                unique_classes, counts = torch.unique(flat_mask, return_counts=True)
                
                # Set 1 for classes that appear in the mask
                for cls_idx, count in zip(unique_classes, counts):
                    if cls_idx < num_classes and count > 0:
                        class_labels[i, cls_idx] = 1
                        
                # If no valid classes found, use class 0
                if class_labels[i].sum() == 0:
                    class_labels[i, 0] = 1
                    
        elif labels.dim() == 4:  # One-hot segmentation masks [batch_size, C, H, W]
            # Extract class presence from one-hot masks
            class_labels = torch.zeros(batch_size, num_classes, device=self.device)
            for i in range(batch_size):
                for c in range(min(labels.size(1), num_classes)):
                    if torch.any(labels[i, c] > 0):
                        class_labels[i, c] = 1
                        
                # If no valid classes found, use class 0
                if class_labels[i].sum() == 0:
                    class_labels[i, 0] = 1
        else:
            print(f"Error: Unsupported label format in weak supervision loss: {labels.shape}")
            # Default to first class
            class_labels = torch.zeros(batch_size, num_classes, device=self.device)
            class_labels[:, 0] = 1
        
        # Now calculate Dice loss with properly formatted labels
        for i in range(batch_size):
            seg_map = segmentation_maps[i]
            
            # Get the target class indices (non-zero elements in class_labels)
            target_indices = torch.nonzero(class_labels[i]).squeeze(1)
            
            # If no target indices, use class 0
            if target_indices.numel() == 0:
                target_indices = torch.tensor([0], device=self.device)
            
            # Calculate Dice loss for each target class
            class_dice_loss = 0
            for idx in target_indices:
                # Create a binary mask for the target class
                target_mask = torch.zeros_like(seg_map)
                target_mask[idx] = 1
                
                # Calculate Dice loss for the target class
                intersection = torch.sum(seg_map[idx] * target_mask[idx])
                union = torch.sum(seg_map[idx]) + torch.sum(target_mask[idx])
                dice_loss = 1 - (2 * intersection + 1e-6) / (union + 1e-6)
                class_dice_loss += dice_loss
            
            # Average over target classes
            loss += class_dice_loss / len(target_indices)
            
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