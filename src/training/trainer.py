import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

class Trainer:
    def __init__(self, model, dataset, config):
        self.model = model
        self.dataset = dataset
        self.config = config
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        
    def train(self):
        for epoch in range(self.config['epochs']):
            self.model.train()
            epoch_loss = 0
            
            print(f'Epoch {epoch+1}/{self.config["epochs"]}')
            for batch_idx, batch in enumerate(self.train_loader):
                # Move data to device
                images = batch['image'].to(self.device)
                labels = batch['mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                logits = outputs['logits']
                segmentation_maps = outputs['segmentation_maps']
                
                # Calculate classification loss
                cls_loss = self.cls_criterion(logits, labels)
                
                # Calculate segmentation loss using weak supervision
                # We use the class activation maps as pseudo-labels
                seg_loss = self._calculate_weak_supervision_loss(
                    segmentation_maps,
                    labels
                )
                
                # Total loss
                loss = cls_loss + self.config['seg_loss_weight'] * seg_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Update progress
                epoch_loss += loss.item()
                
                # Print progress
                if batch_idx % 10 == 0:
                    print(f'Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}')
            
            # Calculate average epoch loss
            avg_epoch_loss = epoch_loss / len(self.train_loader)
            print(f'Epoch {epoch+1} average loss: {avg_epoch_loss:.4f}')
            
            # Save checkpoint
            if (epoch + 1) % self.config['save_interval'] == 0:
                self._save_checkpoint(epoch, avg_epoch_loss)
    
    def _calculate_weak_supervision_loss(self, segmentation_maps, labels):
        """Calculate weak supervision loss using class activation maps"""
        batch_size = segmentation_maps.size(0)
        loss = 0
        
        for i in range(batch_size):
            # Get CAM for the correct class
            cam = segmentation_maps[i, labels[i]]
            
            # Normalize CAM
            cam = torch.relu(cam)
            cam = cam / (cam.max() + 1e-8)
            
            # Calculate loss to encourage high activation in foreground
            # and low activation in background
            loss += -torch.mean(cam)
        
        return loss / batch_size
    
    def _save_checkpoint(self, epoch, loss):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss
        }
        
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
        torch.save(checkpoint, checkpoint_path)
        print(f'Saved checkpoint to {checkpoint_path}') 