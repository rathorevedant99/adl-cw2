"""
Weakly-supervised training with integrated additional data sources.
Implements a training pipeline that leverages original and additional data.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from pathlib import Path
import numpy as np
import random
import time
import os
import json

from src.data_integration import create_integrated_dataset, ConsistencyLoss

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WeaklySupervisedTrainer:
    """
    Trainer for weakly-supervised learning with additional data integration.
    """
    
    def __init__(self, model, original_dataset, additional_data_dir=None, config=None):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            original_dataset: Original dataset with strong supervision
            additional_data_dir: Directory with additional weakly-labeled and unlabeled data
            config: Training configuration
        """
        self.model = model
        self.original_dataset = original_dataset
        self.additional_data_dir = Path(additional_data_dir) if additional_data_dir else None
        self.config = config or {}
        
        # Setup device
        self.device = torch.device(self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        logger.info(f"Using device: {self.device}")
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Training parameters
        self.num_epochs = self.config.get('num_epochs', 50)
        self.batch_size = self.config.get('batch_size', 16)
        self.learning_rate = self.config.get('learning_rate', 1e-4)
        self.weight_decay = self.config.get('weight_decay', 1e-5)
        
        # Additional data parameters
        self.use_additional_data = additional_data_dir is not None
        self.generate_pseudo_labels = self.config.get('generate_pseudo_labels', True)
        self.consistency_weight = self.config.get('consistency_weight', 1.0)
        self.curriculum_learning = self.config.get('curriculum_learning', True)
        
        # Strong augmentations for consistency regularization
        self.strong_augment = self.config.get('strong_augment', False)
        
        # Checkpointing
        self.checkpoint_dir = Path(self.config.get('checkpoint_dir', 'experiments/checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_frequency = self.config.get('save_frequency', 5)
        
        # Create dataset and dataloaders
        self._setup_datasets()
        
        # Setup optimizer and learning rate scheduler
        self._setup_optimization()
        
        # Setup losses
        self._setup_losses()
        
        # Training state
        self.current_epoch = 0
        self.best_validation_score = -1.0
        self.training_losses = []
        self.validation_metrics = []
    
    def _setup_datasets(self):
        """Setup datasets and dataloaders."""
        if self.use_additional_data:
            logger.info("Creating integrated dataset with additional data...")
            self.integrated_dataset = create_integrated_dataset(
                self.original_dataset,
                self.additional_data_dir,
                transform=None,  # Original dataset should already have transforms
                model=self.model if self.generate_pseudo_labels else None,
                device=self.device,
                generate_pseudo=self.generate_pseudo_labels
            )
            
            self.train_dataset = self.integrated_dataset
            
            # Create a separate validation set from the original dataset
            # Assuming original dataset has a split function or similar
            # This would depend on the original dataset's implementation
            if hasattr(self.original_dataset, 'split'):
                # If the dataset has a split method, use it
                self.train_original, self.validation_dataset = self.original_dataset.split(0.8)
            else:
                # Otherwise, use 80% for training and 20% for validation
                total_size = len(self.original_dataset)
                train_size = int(0.8 * total_size)
                val_size = total_size - train_size
                
                indices = list(range(total_size))
                random.shuffle(indices)
                
                train_indices = indices[:train_size]
                val_indices = indices[train_size:]
                
                from torch.utils.data import Subset
                self.train_original = Subset(self.original_dataset, train_indices)
                self.validation_dataset = Subset(self.original_dataset, val_indices)
        else:
            # Just use the original dataset
            # Split into train and validation
            if hasattr(self.original_dataset, 'split'):
                self.train_dataset, self.validation_dataset = self.original_dataset.split(0.8)
            else:
                total_size = len(self.original_dataset)
                train_size = int(0.8 * total_size)
                val_size = total_size - train_size
                
                indices = list(range(total_size))
                random.shuffle(indices)
                
                train_indices = indices[:train_size]
                val_indices = indices[train_size:]
                
                from torch.utils.data import Subset
                self.train_dataset = Subset(self.original_dataset, train_indices)
                self.validation_dataset = Subset(self.original_dataset, val_indices)
            
            self.train_original = self.train_dataset
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )
        
        self.validation_loader = DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )
        
        logger.info(f"Created dataloaders with {len(self.train_loader)} training batches and {len(self.validation_loader)} validation batches")
    
    def _setup_optimization(self):
        """Setup optimizer and learning rate scheduler."""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
    
    def _setup_losses(self):
        """Setup loss functions for training."""
        # Get model-specific loss functions
        if hasattr(self.model, 'get_loss_functions'):
            self.loss_functions = self.model.get_loss_functions()
        else:
            # Default losses
            self.loss_functions = {
                'classification_loss': torch.nn.CrossEntropyLoss(),
                'segmentation_loss': torch.nn.CrossEntropyLoss()
            }
        
        # Add consistency loss for unlabeled data
        self.consistency_loss = ConsistencyLoss(weight=self.consistency_weight)
    
    def train(self):
        """Train the model."""
        logger.info(f"Starting training for {self.num_epochs} epochs")
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            logger.info(f"Epoch {epoch+1}/{self.num_epochs}")
            
            # Update curriculum if using integrated dataset
            if self.use_additional_data and self.curriculum_learning:
                self.integrated_dataset.update_curriculum()
            
            # Train one epoch
            train_loss, train_metrics = self._train_epoch()
            
            # Validate
            val_loss, val_metrics = self._validate()
            
            # Update learning rate
            val_score = val_metrics.get('miou', 0.0)
            self.scheduler.step(val_score)
            
            # Track best model
            if val_score > self.best_validation_score:
                self.best_validation_score = val_score
                self._save_checkpoint(is_best=True)
                logger.info(f"New best model with validation mIoU: {val_score:.4f}")
            
            # Save regular checkpoint
            if (epoch + 1) % self.save_frequency == 0:
                self._save_checkpoint()
            
            # Log metrics
            self._log_metrics(train_metrics, val_metrics, train_loss, val_loss)
            
            # Generate new pseudo-labels periodically
            if self.use_additional_data and self.generate_pseudo_labels and epoch > 0 and epoch % 10 == 0:
                logger.info("Regenerating pseudo-labels...")
                self._regenerate_pseudo_labels()
        
        logger.info("Training completed")
        
        # Return training history
        return {
            'training_losses': self.training_losses,
            'validation_metrics': self.validation_metrics,
            'best_validation_score': self.best_validation_score
        }
    
    def _train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        metrics = {'classification_acc': 0.0}
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Get images and targets
            images = batch['image'].to(self.device)
            
            # Check for labels
            if 'label' in batch:
                labels = batch['label'].to(self.device)
                has_label = True
            else:
                has_label = False
            
            # Check for segmentation masks
            if 'mask' in batch:
                masks = batch['mask'].to(self.device)
                has_mask = True
            else:
                has_mask = False
            
            # Check for dataset source
            dataset_source = batch.get('dataset_source', ['original'] * images.shape[0])
            
            # Apply strong augmentation for consistency regularization if enabled
            if self.strong_augment:
                # Create a strongly augmented version of the image
                # This depends on your specific augmentation implementation
                augmented_images = self._apply_strong_augmentation(images)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images, labels=labels if has_label else None)
            
            # Calculate losses
            loss = 0.0
            
            # Classification loss (for images with labels)
            if has_label:
                classification_loss = self.loss_functions['classification_loss'](outputs['logits'], labels.argmax(dim=1) if labels.dim() > 1 else labels)
                loss += classification_loss
                
                # For metrics
                pred_classes = outputs['logits'].argmax(dim=1)
                target_classes = labels.argmax(dim=1) if labels.dim() > 1 else labels
                accuracy = (pred_classes == target_classes).float().mean().item()
                metrics['classification_acc'] += accuracy
            
            # Segmentation loss (for images with masks)
            if has_mask:
                segmentation_loss = self.loss_functions.get('segmentation_loss', torch.nn.CrossEntropyLoss())(
                    outputs['segmentation_maps'], masks.squeeze(1).long()
                )
                loss += segmentation_loss
            
            # Size constraint loss (to prevent degenerate segmentations)
            if 'size_constraint_loss' in self.loss_functions and has_label:
                size_loss = self.loss_functions['size_constraint_loss'](
                    outputs['segmentation_maps'], labels
                )
                loss += size_loss
            
            # Consistency loss (for unlabeled data or with strong augmentation)
            if self.strong_augment:
                # Get predictions from original and augmented images
                with torch.no_grad():
                    aug_outputs = self.model(augmented_images)
                
                # Calculate consistency loss between the two predictions
                consistency_loss = self.consistency_loss(
                    outputs['logits'], aug_outputs['logits']
                )
                loss += consistency_loss
            
            # CAM consistency loss (enforces consistency between CAM and segmentation)
            if 'consistency_loss' in self.loss_functions:
                cam_consistency_loss = self.loss_functions['consistency_loss'](
                    outputs['cam_maps'], outputs['segmentation_maps']
                )
                loss += cam_consistency_loss
            
            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
            
            # Update running loss
            running_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Calculate average loss and metrics
        avg_loss = running_loss / len(self.train_loader)
        metrics['classification_acc'] /= len(self.train_loader)
        
        return avg_loss, metrics
    
    def _validate(self):
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.validation_loader, desc="Validating"):
                # Get images and targets
                images = batch['image'].to(self.device)
                
                if 'label' in batch:
                    labels = batch['label'].to(self.device)
                else:
                    continue  # Skip if no labels
                
                if 'mask' in batch:
                    masks = batch['mask'].to(self.device)
                    has_mask = True
                else:
                    has_mask = False
                
                # Forward pass
                outputs = self.model(images, is_training=False)
                
                # Calculate classification loss
                classification_loss = self.loss_functions['classification_loss'](
                    outputs['logits'], labels.argmax(dim=1) if labels.dim() > 1 else labels
                )
                loss = classification_loss
                
                # Calculate segmentation metrics if masks available
                if has_mask:
                    # Get predictions
                    pred_masks = outputs['segmentation_maps'].argmax(dim=1).cpu().numpy()
                    true_masks = masks.squeeze(1).cpu().numpy()
                    
                    # Store for metric calculation
                    all_preds.extend(pred_masks)
                    all_targets.extend(true_masks)
                
                # Update running loss
                running_loss += loss.item()
        
        # Calculate average loss
        avg_loss = running_loss / len(self.validation_loader)
        
        # Calculate metrics
        metrics = {}
        
        # Calculate IoU if we have masks
        if all_preds:
            metrics['miou'] = self._calculate_miou(all_preds, all_targets)
        
        return avg_loss, metrics
    
    def _calculate_miou(self, preds, targets):
        """Calculate mean IoU between predicted and target masks."""
        num_classes = self.config.get('num_classes', 3)  # Default: 3 (cat, dog, background)
        
        # Initialize IoU for each class
        ious = []
        
        for cls in range(num_classes):
            # Convert to binary masks
            pred_binary = (np.array(preds) == cls).astype(np.int32)
            target_binary = (np.array(targets) == cls).astype(np.int32)
            
            # Calculate intersection and union
            intersection = np.logical_and(pred_binary, target_binary).sum()
            union = np.logical_or(pred_binary, target_binary).sum()
            
            # Calculate IoU
            iou = intersection / (union + 1e-8)  # Add small epsilon to avoid division by zero
            ious.append(iou)
        
        # Return mean IoU
        return np.mean(ious)
    
    def _save_checkpoint(self, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_validation_score': self.best_validation_score,
            'training_losses': self.training_losses,
            'validation_metrics': self.validation_metrics
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch+1}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
    
    def _log_metrics(self, train_metrics, val_metrics, train_loss, val_loss):
        """Log training and validation metrics."""
        log_str = (f"Epoch {self.current_epoch+1}/{self.num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}")
        
        # Add metrics to log string
        for metric, value in train_metrics.items():
            log_str += f", Train {metric}: {value:.4f}"
        
        for metric, value in val_metrics.items():
            log_str += f", Val {metric}: {value:.4f}"
        
        logger.info(log_str)
        
        # Track metrics
        self.training_losses.append({
            'epoch': self.current_epoch,
            'loss': train_loss,
            'metrics': train_metrics
        })
        
        self.validation_metrics.append({
            'epoch': self.current_epoch,
            'loss': val_loss,
            'metrics': val_metrics
        })
        
        # Save metrics to file
        metrics_path = self.checkpoint_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump({
                'training': self.training_losses,
                'validation': self.validation_metrics
            }, f, indent=2)
    
    def _regenerate_pseudo_labels(self):
        """Regenerate pseudo-labels with current model."""
        if not self.use_additional_data or not self.generate_pseudo_labels:
            return
        
        from src.data_integration import PseudoLabeler, AdditionalDataset
        
        # Create dataset for unlabeled data
        unlabeled_dataset = AdditionalDataset(
            self.additional_data_dir,
            transform=None,  # Dataset should handle transforms
            label_type='none'
        )
        
        # Generate pseudo-labels
        pseudo_label_dir = self.additional_data_dir / 'pseudo_labels'
        
        # Create pseudo-labeler
        labeler = PseudoLabeler(self.model, self.device)
        
        # Generate pseudo-labels
        pseudo_labels = labeler.generate_pseudo_labels(unlabeled_dataset, pseudo_label_dir)
        
        # Reinitialize the integrated dataset with new pseudo-labels
        self._setup_datasets()
        
        logger.info(f"Regenerated {len(pseudo_labels)} pseudo-labels")
    
    def _apply_strong_augmentation(self, images):
        """Apply strong augmentation to images for consistency regularization."""
        # This is a placeholder - implement your specific augmentations
        # For example, random color jitter, grayscale, etc.
        
        # Simple augmentation: random color jitter
        from torchvision import transforms
        
        color_jitter = transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
        )
        
        augmented_images = []
        for image in images:
            # Apply augmentation
            aug_image = color_jitter(image)
            augmented_images.append(aug_image)
        
        return torch.stack(augmented_images)


def train_model_with_additional_data(model, original_dataset, additional_data_dir, config=None):
    """
    Train a model with additional weakly-labeled and unlabeled data.
    
    Args:
        model: Model to train
        original_dataset: Original dataset
        additional_data_dir: Directory with additional data
        config: Training configuration
        
    Returns:
        Trained model and training metrics
    """
    trainer = WeaklySupervisedTrainer(
        model=model,
        original_dataset=original_dataset,
        additional_data_dir=additional_data_dir,
        config=config
    )
    
    training_history = trainer.train()
    
    return trainer.model, training_history
