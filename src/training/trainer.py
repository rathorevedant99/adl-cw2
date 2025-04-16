import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import logging
from torch.amp import autocast, GradScaler

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
        
        # Enable CUDA optimizations if using CUDA
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
        
        # Setup data loader with pinned memory if enabled
        self.train_loader = DataLoader(
            dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['training']['num_workers'],
            pin_memory=config['training'].get('pin_memory', False)
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
        
        # Setup automatic mixed precision
        self.use_amp = config['training'].get('amp', False)
        self.scaler = GradScaler() if self.use_amp else None
        
        logging.info(f"Training configuration: {self.config}")
        
    def train(self):
        for epoch in range(self.config['training']['num_epochs']):
            self.model.train()
            epoch_loss = 0
            
            logging.info(f'Epoch {epoch+1}/{self.config["training"]["num_epochs"]}')
            for batch_idx, batch in enumerate(self.train_loader):
                images = batch['image'].to(self.device)
                labels = batch['mask'].to(self.device)
                
                # Use automatic mixed precision if enabled
                if self.use_amp:
                    with autocast(device_type=self.device.type):
                        outputs = self.model(images)
                        logits = outputs['logits']
                        segmentation_maps = outputs['segmentation_maps']
                        
                        cls_loss = self.cls_criterion(logits, labels)
                        seg_loss = self._calculate_weak_supervision_loss(segmentation_maps, labels)
                        loss = cls_loss + self.seg_loss_weight * seg_loss
                    
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(images)
                    logits = outputs['logits']
                    segmentation_maps = outputs['segmentation_maps']
                    
                    cls_loss = self.cls_criterion(logits, labels)
                    seg_loss = self._calculate_weak_supervision_loss(segmentation_maps, labels)
                    loss = cls_loss + self.seg_loss_weight * seg_loss
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                
                epoch_loss += loss.item()
                
                # Log progress
                if batch_idx % 10 == 0:
                    logging.info(f'Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}')
            
            # Calculate average epoch loss
            avg_epoch_loss = epoch_loss / len(self.train_loader)
            logging.info(f'Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}')
            
            # Save checkpoint
            if (epoch + 1) % self.save_interval == 0:
                self._save_checkpoint(epoch + 1, avg_epoch_loss)
                
    def _calculate_weak_supervision_loss(self, segmentation_maps, labels):
        """Calculate weak supervision loss for segmentation"""
        batch_size = segmentation_maps.size(0)
        loss = 0
        
        for i in range(batch_size): # Dice loss. Could change to other loss functions.
            seg_map = segmentation_maps[i]
            label = labels[i]
            intersection = torch.sum(seg_map * label)
            union = torch.sum(seg_map) + torch.sum(label)
            dice_loss = 1 - (2 * intersection + 1e-6) / (union + 1e-6)
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