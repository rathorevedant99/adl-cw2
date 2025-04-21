import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import logging
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import optuna
from torch.cuda.amp import autocast, GradScaler

class Trainer:
    def __init__(self, model, train_dataset, val_dataset, config, weak_supervision_types=None, optuna_trial=None):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.weak_supervision_types = weak_supervision_types or ['labels']
        # optuna Trial for pruning (None when not tuning)
        self.optuna_trial = optuna_trial
        
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

        # Setup data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['training']['num_workers'],
            pin_memory=config['training'].get('pin_memory', False)
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['training']['num_workers'],
            pin_memory=config['training'].get('pin_memory', False)
        )
        
        # Setup optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Mixed-precision scaler
        self.use_amp = config['training'].get('amp', False)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Setup loss functions and training method
        self.checkpoint_dir = Path(config['training']['checkpoint_dir'])
        self.log_dir = Path(config['training']['log_dir'])
        self.method = config['model'].get('method', 'WS').upper()
        
        if self.method not in ('WS', 'FS'):
            raise ValueError(f"Unsupported method: {self.method}")
            
        if self.method == 'FS':
            # fully‑supervised: pixel‐wise CE
            self.seg_criterion = nn.CrossEntropyLoss()
        else:
            # Weakly supervised approach
            self.cls_criterion = nn.CrossEntropyLoss()
            self.seg_loss_weight = config['training']['seg_loss_weight']
            self.apply_region_growing = config['training'].get('apply_region_growing', True)
        
        self.save_interval = config['training']['save_interval']
        
        # For tracking loss
        self.epoch_train_losses = []
        self.epoch_val_losses = []
        self.batch_losses = []
        
        # Track supervision info for plotting
        self.supervised_with = '_'.join(self.weak_supervision_types)

        logging.debug(f"Training configuration: {self.config}")
        logging.info(f"Training with weak supervision types: {self.weak_supervision_types}")
        
        # Log AMP status
        if self.use_amp:
            logging.info("Using Automatic Mixed Precision (AMP)")

    def train(self):
        for epoch in range(self.config['training']['num_epochs']):
            self.model.train()
            epoch_loss = 0
            epoch_batch_losses = []

            logging.info(f'Epoch {epoch+1}/{self.config["training"]["num_epochs"]}')
            for batch_idx, batch in enumerate(self.train_loader):
                images = batch['image'].to(self.device)
                labels = batch['mask'].to(self.device)

                # Debug labels in first epoch
                if epoch == 0 and batch_idx < 2:
                    logging.info(f"Label shape: {labels.shape}")
                    logging.info(f"Label dtype: {labels.dtype}")
                    if labels.numel() < 100:  # Only log all values if there aren't too many
                        logging.info(f"Label values: {labels}")
                    if len(labels.shape) == 1:
                        logging.info("CONFIRMATION: Labels are scalar values (one per image)")
                    else:
                        logging.info(f"Labels have shape {labels.shape} - they appear to be masks, not scalars")
                
                # Forward pass with or without mixed precision
                self.optimizer.zero_grad()
                
                if self.use_amp:
                    with autocast():
                        loss = self._forward_pass(images, labels)
                        
                    # Backward pass with scaler
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Standard forward/backward without mixed precision
                    loss = self._forward_pass(images, labels)
                    loss.backward()
                    self.optimizer.step()

                # Track loss
                batch_loss = loss.item()
                epoch_loss += batch_loss
                epoch_batch_losses.append(batch_loss)
                self.batch_losses.append(batch_loss)

                if batch_idx % 10 == 0:
                    logging.info(f'Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}')

            avg_train_loss = epoch_loss / len(self.train_loader)
            self.epoch_train_losses.append(avg_train_loss)
            logging.info(f'Epoch {epoch+1} training loss: {avg_train_loss:.4f}')

            # Validation
            val_loss = self._evaluate_on_validation_set()
            self.epoch_val_losses.append(val_loss)
            logging.info(f'Epoch {epoch+1} validation loss: {val_loss:.4f}')                    
            
            # Report to Optuna if we're in a trial
            if self.optuna_trial is not None:
                # report intermediate objective
                self.optuna_trial.report(val_loss, epoch)
                # ask to prune?
                if self.optuna_trial.should_prune():
                    logging.info(f"Pruning trial at epoch {epoch+1}")
                    raise optuna.TrialPruned()

            # Save checkpoint
            if (epoch + 1) % self.save_interval == 0:
                self._save_checkpoint(epoch + 1, avg_train_loss)

        # Plot loss after training completes
        self._plot_loss()
    
    def _forward_pass(self, images, labels):
        """Forward pass computation based on training method"""
        if self.method == 'WS':
            # Weakly-supervised approach
            # Prepare labels for region growing if needed
            if self.apply_region_growing:
                if 'scribbles' in self.weak_supervision_types and hasattr(labels, 'dim') and labels.dim() > 1:
                    # For scribble masks, use directly
                    one_hot_labels = labels
                elif 'bboxes' in self.weak_supervision_types and hasattr(labels, 'dim') and labels.dim() > 1:
                    # For bbox masks, use directly
                    one_hot_labels = labels
                else:
                    # For class labels, convert to one-hot
                    one_hot_labels = self._to_one_hot(labels)
            else:
                one_hot_labels = None

            # Forward pass with region growing
            outputs = self.model(images, labels=one_hot_labels, apply_region_growing=self.apply_region_growing)
            logits = outputs['logits']
            segmentation_maps = outputs['segmentation_maps']

            # Classification loss
            cls_loss = self.cls_criterion(logits, labels if labels.dim() == 1 else torch.zeros(labels.size(0), device=self.device).long())

            # Weak supervision loss
            if 'scribbles' in self.weak_supervision_types and labels.dim() > 1:
                # Pixel-level supervision for scribbles
                seg_loss = self._calculate_pixel_supervision_loss(segmentation_maps, labels)
            elif 'bboxes' in self.weak_supervision_types and labels.dim() > 1:
                # Pixel-level supervision for bounding boxes
                seg_loss = self._calculate_pixel_supervision_loss(segmentation_maps, labels)
            else:
                # Regular weak supervision loss for class labels
                seg_loss = self._calculate_weak_supervision_loss(segmentation_maps, labels)

            # Total loss
            loss = cls_loss + self.seg_loss_weight * seg_loss
        else:
            # Fully supervised approach
            outputs = self.model(images)
            seg_maps = outputs['segmentation_maps']      # [B, C, H, W]
            # standard pixel‐wise cross‐entropy
            loss = self.seg_criterion(seg_maps, labels)  # masks: [B, H, W] with 0..C‑1  
        
        return loss

    def _evaluate_on_validation_set(self):
        """Evaluate the model on the validation set"""
        self.model.eval()
        total_loss = 0
        count = 0
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                labels = batch['mask'].to(self.device)
                
                if self.use_amp:
                    with autocast():
                        if self.method == 'WS':
                            # Prepare labels for region growing if needed
                            if self.apply_region_growing:
                                if 'scribbles' in self.weak_supervision_types and hasattr(labels, 'dim') and labels.dim() > 1:
                                    one_hot_labels = labels
                                elif 'bboxes' in self.weak_supervision_types and hasattr(labels, 'dim') and labels.dim() > 1:
                                    one_hot_labels = labels
                                else:
                                    one_hot_labels = self._to_one_hot(labels)
                            else:
                                one_hot_labels = None
                                
                            # Forward pass
                            outputs = self.model(
                                images,
                                labels=one_hot_labels,
                                apply_region_growing=self.apply_region_growing
                            )
                            
                            # Calculate losses
                            cls_loss = self.cls_criterion(outputs['logits'], 
                                                     labels if labels.dim() == 1 else torch.zeros(labels.size(0), device=self.device).long())
                            
                            if 'scribbles' in self.weak_supervision_types and labels.dim() > 1:
                                seg_loss = self._calculate_pixel_supervision_loss(outputs['segmentation_maps'], labels)
                            elif 'bboxes' in self.weak_supervision_types and labels.dim() > 1:
                                seg_loss = self._calculate_pixel_supervision_loss(outputs['segmentation_maps'], labels)
                            else:
                                seg_loss = self._calculate_weak_supervision_loss(outputs['segmentation_maps'], labels)
                                
                            loss = cls_loss + self.seg_loss_weight * seg_loss
                        else:
                            # Fully supervised mode
                            outputs = self.model(images)
                            loss = self.seg_criterion(outputs['segmentation_maps'], labels)
                else:
                    # No mixed precision
                    if self.method == 'WS':
                        # Prepare labels for region growing if needed
                        if self.apply_region_growing:
                            if 'scribbles' in self.weak_supervision_types and hasattr(labels, 'dim') and labels.dim() > 1:
                                one_hot_labels = labels
                            elif 'bboxes' in self.weak_supervision_types and hasattr(labels, 'dim') and labels.dim() > 1:
                                one_hot_labels = labels
                            else:
                                one_hot_labels = self._to_one_hot(labels)
                        else:
                            one_hot_labels = None
                            
                        # Forward pass
                        outputs = self.model(
                            images,
                            labels=one_hot_labels,
                            apply_region_growing=self.apply_region_growing
                        )
                        
                        # Calculate losses
                        cls_loss = self.cls_criterion(outputs['logits'], 
                                                 labels if labels.dim() == 1 else torch.zeros(labels.size(0), device=self.device).long())
                        
                        if 'scribbles' in self.weak_supervision_types and labels.dim() > 1:
                            seg_loss = self._calculate_pixel_supervision_loss(outputs['segmentation_maps'], labels)
                        elif 'bboxes' in self.weak_supervision_types and labels.dim() > 1:
                            seg_loss = self._calculate_pixel_supervision_loss(outputs['segmentation_maps'], labels)
                        else:
                            seg_loss = self._calculate_weak_supervision_loss(outputs['segmentation_maps'], labels)
                            
                        loss = cls_loss + self.seg_loss_weight * seg_loss
                    else:
                        # Fully supervised mode
                        outputs = self.model(images)
                        loss = self.seg_criterion(outputs['segmentation_maps'], labels)

                total_loss += loss.item()
                count += 1

        return total_loss / count if count > 0 else 0.0

    def _calculate_weak_supervision_loss(self, segmentation_maps, labels):
        """Calculate weak supervision loss using pseudo masks"""
        batch_size = segmentation_maps.size(0)
        loss = 0

        for i in range(batch_size):
            seg_map = segmentation_maps[i]
            label_idx = labels[i].item()
            target_activation = seg_map[label_idx]
            target_activation = torch.sigmoid(target_activation)

            threshold = 0.5
            binary_mask = (target_activation > threshold).float()

            intersection = (target_activation * binary_mask).sum()
            union = target_activation.sum() + binary_mask.sum()
            dice_score = (2 * intersection + 1e-6) / (union + 1e-6)
            dice_loss = 1 - dice_score

            loss += dice_loss

        return loss / batch_size
    
    def _calculate_pixel_supervision_loss(self, segmentation_maps, masks):
        """
        Calculate loss for pixel-level supervision (bboxes or scribbles)
        
        Args:
            segmentation_maps: Model segmentation output [B, C, H, W]
            masks: Supervision masks [B, H, W]
            
        Returns:
            loss: Segmentation loss
        """
        batch_size = segmentation_maps.size(0)
        num_classes = segmentation_maps.size(1)
        
        # Initialize loss
        loss = 0
        
        # If masks are 2D (B, H, W), we need to filter valid pixels for each batch and class
        if masks.dim() == 3:
            for b in range(batch_size):
                # Get the valid pixels mask
                valid_mask = masks[b] > 0
                if valid_mask.sum() == 0:
                    continue
                
                # For each class in the segmentation map
                for c in range(num_classes):
                    # Get class pixels in the valid mask
                    class_pixels = masks[b] == c
                    if class_pixels.sum() == 0:
                        continue
                    
                    # Get segmentation outputs for this class
                    seg_out = segmentation_maps[b, c]
                    
                    # Calculate binary cross-entropy loss for valid pixels
                    target = (masks[b] == c).float()
                    pred = torch.sigmoid(seg_out)
                    
                    # BCE loss on valid pixels only
                    bce_loss = nn.functional.binary_cross_entropy(
                        pred[valid_mask], 
                        target[valid_mask], 
                        reduction='mean'
                    )
                    loss += bce_loss
        # If masks are already one-hot encoded (B, C, H, W)
        elif masks.dim() == 4:
            # Calculate per-pixel binary cross-entropy
            for c in range(num_classes):
                valid_mask = masks[:, c].sum(dim=(1, 2)) > 0
                if valid_mask.sum() == 0:
                    continue
                
                target = masks[valid_mask, c]
                pred = torch.sigmoid(segmentation_maps[valid_mask, c])
                bce_loss = nn.functional.binary_cross_entropy(
                    pred, target, reduction='mean'
                )
                loss += bce_loss
                
        return loss / batch_size

    def _to_one_hot(self, labels):
        """Convert class indices to one-hot encoded labels"""
        batch_size = labels.size(0)
        num_classes = self.model.num_classes if hasattr(self.model, 'num_classes') else self.config['model']['num_classes']
        one_hot = torch.zeros(batch_size, num_classes, device=self.device)
        one_hot[torch.arange(batch_size), labels] = 1.0
        return one_hot

    def _save_checkpoint(self, epoch, loss):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'config': self.config,
            'weak_supervision_types': self.weak_supervision_types
        }

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        logging.info(f'Saved checkpoint to {checkpoint_path}')
    
    def _plot_loss(self):
        """Plot both training and validation loss using Pillow"""
        width, height = 1000, 500
        margin = 60
        plot_width = width - 2 * margin
        plot_height = height - 2 * margin

        image = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(image)

        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except IOError:
            font = ImageFont.load_default()

        # Draw axes
        draw.rectangle([(margin, margin), (width - margin, height - margin)], outline='black')

        # Axis labels
        draw.text((width // 2, height - margin // 2), "Epochs", fill='black', anchor="mm", font=font)
        draw.text((margin // 2, height // 2), "Loss", fill='black', anchor="mm", font=font, angle=90)
        draw.text((width // 2, margin // 2), f"Training and Validation Loss ({self.supervised_with})", fill='black', anchor="mm", font=font)

        if self.epoch_train_losses:
            train_losses = np.array(self.epoch_train_losses)
            val_losses = np.array(self.epoch_val_losses)
            all_losses = np.concatenate([train_losses, val_losses])
            min_loss, max_loss = all_losses.min(), all_losses.max()
            loss_range = max_loss - min_loss
            if loss_range == 0:
                loss_range = max_loss * 0.1 or 0.1
            min_loss -= loss_range * 0.05
            max_loss += loss_range * 0.05

            def normalize(losses):
                return [
                    margin + plot_height - plot_height * (loss - min_loss) / (max_loss - min_loss)
                    for loss in losses
                ]

            train_y = normalize(train_losses)
            val_y = normalize(val_losses)
            x_step = plot_width / (len(train_losses) - 1) if len(train_losses) > 1 else plot_width
            x_coords = [margin + i * x_step for i in range(len(train_losses))]

            # Draw lines
            for i in range(1, len(x_coords)):
                draw.line([x_coords[i - 1], train_y[i - 1], x_coords[i], train_y[i]], fill='blue', width=2)
                draw.line([x_coords[i - 1], val_y[i - 1], x_coords[i], val_y[i]], fill='green', width=2)

            # Draw dots
            for x, y in zip(x_coords, train_y):
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill='blue')
            for x, y in zip(x_coords, val_y):
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill='green')

            # Axis ticks
            for i in range(6):
                y_pos = margin + plot_height - i * plot_height / 5
                tick_value = min_loss + i * (max_loss - min_loss) / 5
                draw.text((margin - 10, y_pos), f"{tick_value:.4f}", fill='black', anchor="rm", font=font)

            for i in range(min(len(train_losses), 10)):
                epoch = i * (len(train_losses) - 1) // 9 if len(train_losses) > 1 else 0
                x_pos = margin + epoch * x_step
                draw.text((x_pos, height - margin + 10), str(epoch), fill='black', anchor="mt", font=font)

            # Legend
            legend_x = width - margin - 150
            legend_y = margin + 10
            draw.text((legend_x, legend_y), "Legend:", fill='black', font=font)
            draw.line([(legend_x, legend_y + 20), (legend_x + 20, legend_y + 20)], fill='blue', width=2)
            draw.text((legend_x + 30, legend_y + 20), "Train Loss", fill='black', font=font)
            draw.line([(legend_x, legend_y + 40), (legend_x + 20, legend_y + 40)], fill='green', width=2)
            draw.text((legend_x + 30, legend_y + 40), "Val Loss", fill='black', font=font)

        self.log_dir.mkdir(parents=True, exist_ok=True)
        plot_dir = self.log_dir / "plots"
        plot_dir.mkdir(exist_ok=True)
        
        image.save(plot_dir / f"loss_plot_{self.supervised_with}.png")
        logging.info(f"Loss plot saved to {plot_dir / f'loss_plot_{self.supervised_with}.png'}")