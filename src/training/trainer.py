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
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers']
        )
        
        # Setup optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Setup loss functions
        self.cls_criterion = nn.CrossEntropyLoss()
            
        # Setup checkpoint directory
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup log directory
        self.log_dir = Path(config['log_dir'])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup loss weights
        self.seg_loss_weight = config.get('seg_loss_weight', 0.1)
        
        # Setup save interval
        self.save_interval = config.get('save_interval', 5)
        
        logging.info(f"Training configuration: {self.config}")
        
    def train(self):
        for epoch in range(self.config['num_epochs']):
            self.model.train()
            epoch_loss = 0
            
            logging.info(f'Epoch {epoch+1}/{self.config["num_epochs"]}')
            for batch_idx, batch in enumerate(self.train_loader):
                images = batch['image'].to(self.device)
                labels = batch['mask'].to(self.device)
>>>>>>> parent of e95ffbd (Merge pull request #4 from rathorevedant99/vedant-dev)
                
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
        
        for i in range(batch_size):
            # Get the segmentation map for the correct class
            seg_map = segmentation_maps[i, labels[i]]
            
            # Calculate loss based on the segmentation map
            # This is a simple implementation - you might want to modify this
            loss += torch.mean(1 - seg_map)  # Encourage high values in the correct class regions
            
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