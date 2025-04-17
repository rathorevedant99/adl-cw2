import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import logging
import datetime
import os

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
        self.save_interval = config['training']['save_interval']
        
        # For tracking loss (without visualization)
        self.epoch_losses = []
        self.batch_losses = []
        
        logging.debug(f"Training configuration: {self.config}")
        
    def train(self):
        for epoch in range(self.config['training']['num_epochs']):
            self.model.train()
            epoch_loss = 0
            epoch_batch_losses = []
            
            logging.info(f'Epoch {epoch+1}/{self.config["training"]["num_epochs"]}')
            for batch_idx, batch in enumerate(self.train_loader):
                images = batch['image'].to(self.device)
                labels = batch['mask'].to(self.device)
                
                # Debug labels - print this info for the first few batches
                if epoch == 0 and batch_idx < 2:  # Only print for first 2 batches of first epoch
                    logging.info(f"Label shape: {labels.shape}")
                    logging.info(f"Label data type: {labels.dtype}")
                    logging.info(f"Label values: {labels}")
                    
                    # Check if labels are scalars (one value per sample)
                    if len(labels.shape) == 1:
                        logging.info("CONFIRMATION: Labels are scalar values (one per image)")
                    else:
                        logging.info(f"Labels have shape {labels.shape} - they appear to be masks, not scalars")
                
                outputs = self.model(images) # Forward pass
                logits = outputs['logits'] # Required for classification loss
                segmentation_maps = outputs['segmentation_maps'] # Required for segmentation loss
                
                # Calculate classification loss
                cls_loss = self.cls_criterion(logits, labels)
                
                # Calculate weak supervision loss
                seg_loss = self._calculate_weak_supervision_loss(segmentation_maps, labels)
                
                # Total loss
                loss = cls_loss + self.seg_loss_weight * seg_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Track losses
                batch_loss = loss.item()
                epoch_loss += batch_loss
                epoch_batch_losses.append(batch_loss)
                self.batch_losses.append(batch_loss)
                
                # Log progress
                if batch_idx % 10 == 0:
                    logging.info(f'Batch {batch_idx}/{len(self.train_loader)}, Loss: {batch_loss:.4f}')
            
            # Calculate average epoch loss
            avg_epoch_loss = epoch_loss / len(self.train_loader)
            self.epoch_losses.append(avg_epoch_loss)
            logging.info(f'Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}')
            
            # Save checkpoint
            if (epoch + 1) % self.save_interval == 0:
                self._save_checkpoint(epoch + 1, avg_epoch_loss)
                
    def _calculate_weak_supervision_loss(self, segmentation_maps, labels):
        """Calculate weak supervision loss using pseudo masks"""
        batch_size = segmentation_maps.size(0)
        loss = 0
        
        for i in range(batch_size):
            # Get all segmentation maps for this sample
            seg_map = segmentation_maps[i]  # Shape: [num_classes, H, W]
            
            # Get the target class
            label_idx = labels[i].item()  # Convert to Python scalar
            
            # Create a pseudo mask using the activation map for the target class
            target_activation = seg_map[label_idx]  # Shape: [H, W]
            
            # Normalize activation to [0,1] range
            target_activation = torch.sigmoid(target_activation)
            
            # Create a binary pseudo-mask by thresholding
            threshold = 0.5  # You can experiment with different thresholds
            binary_mask = (target_activation > threshold).float()
            
            # Calculate Dice score between activation map and binary mask
            intersection = (target_activation * binary_mask).sum()
            union = target_activation.sum() + binary_mask.sum()
            dice_score = (2 * intersection + 1e-6) / (union + 1e-6)
            dice_loss = 1 - dice_score
            
            loss += dice_loss
        
        return loss / batch_size
    
    def _save_checkpoint(self, epoch, loss):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'config': self.config
        }
        
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        logging.info(f'Saved checkpoint to {checkpoint_path}')