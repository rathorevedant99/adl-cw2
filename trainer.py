import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import logging
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np

class Trainer:
    def __init__(self, model, train_dataset, val_dataset, config, weak_supervision_types=None):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.weak_supervision_types = weak_supervision_types or ['labels']

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
            num_workers=config['training']['num_workers']
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['training']['num_workers']
        )

        logging.info(f"Created DataLoader with batch_size={config['training']['batch_size']}, dataset size={len(train_dataset)}")
        logging.info(f"Expected number of batches: {len(train_dataset) / config['training']['batch_size']:.1f}, actual: {len(self.train_loader)}")

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
        self.apply_region_growing = config['training'].get('apply_region_growing', True)

        # For tracking loss
        self.epoch_train_losses = []
        self.epoch_val_losses = []
        
        # Track metrics by supervision type
        self.supervised_with = '_'.join(self.weak_supervision_types)

        logging.debug(f"Training configuration: {self.config}")
        logging.info(f"Training with weak supervision types: {self.weak_supervision_types}")

    def train(self):
        for epoch in range(self.config['training']['num_epochs']):
            self.model.train()
            epoch_loss = 0

            logging.info(f'Epoch {epoch+1}/{self.config["training"]["num_epochs"]}')
            for batch_idx, batch in enumerate(self.train_loader):
                images = batch['image'].to(self.device)
                
                # Get the appropriate mask based on weak supervision types
                if 'scribbles' in self.weak_supervision_types and 'scribble_mask' in batch:
                    masks = batch['scribble_mask'].to(self.device)
                    is_pixel_mask = True
                elif 'bboxes' in self.weak_supervision_types and 'bbox_mask' in batch:
                    masks = batch['bbox_mask'].to(self.device)
                    is_pixel_mask = True
                else:
                    # Fallback to class labels
                    masks = batch['class_label'].to(self.device)
                    is_pixel_mask = False

                one_hot_labels = None
                if self.apply_region_growing:
                    if is_pixel_mask:
                        # For pixel-level masks, we might need to convert to one-hot
                        # This depends on how your region growing expects labels
                        one_hot_labels = masks
                    else:
                        # For class labels, convert to one-hot
                        one_hot_labels = self._to_one_hot(masks)

                outputs = self.model(images, labels=one_hot_labels, apply_region_growing=self.apply_region_growing)
                logits = outputs['logits']
                segmentation_maps = outputs['segmentation_maps']

                # Class prediction loss always exists
                cls_loss = self.cls_criterion(logits, batch['class_label'].to(self.device))
                
                # Segmentation loss depends on the weak supervision
                if is_pixel_mask:
                    # Pixel-level supervision (bboxes or scribbles)
                    seg_loss = self._calculate_pixel_supervision_loss(segmentation_maps, masks)
                else:
                    # Class-level supervision
                    seg_loss = self._calculate_weak_supervision_loss(segmentation_maps, masks)
                
                loss = cls_loss + self.seg_loss_weight * seg_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

                if batch_idx % 10 == 0:
                    logging.info(f'Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}')

            avg_train_loss = epoch_loss / len(self.train_loader)
            self.epoch_train_losses.append(avg_train_loss)
            logging.info(f'Epoch {epoch+1} training loss: {avg_train_loss:.4f}')

            # Validation
            val_loss = self._evaluate_on_validation_set()
            self.epoch_val_losses.append(val_loss)
            logging.info(f'Epoch {epoch+1} validation loss: {val_loss:.4f}')

            # Save checkpoint
            if (epoch + 1) % self.save_interval == 0:
                self._save_checkpoint(epoch + 1, avg_train_loss)

            # Plot loss after training
            self._plot_loss()

    def _evaluate_on_validation_set(self):
        self.model.eval()
        total_loss = 0
        count = 0

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                
                # Get the appropriate mask based on weak supervision types
                if 'scribbles' in self.weak_supervision_types and 'scribble_mask' in batch:
                    masks = batch['scribble_mask'].to(self.device)
                    is_pixel_mask = True
                elif 'bboxes' in self.weak_supervision_types and 'bbox_mask' in batch:
                    masks = batch['bbox_mask'].to(self.device)
                    is_pixel_mask = True
                else:
                    # Fallback to class labels
                    masks = batch['class_label'].to(self.device)
                    is_pixel_mask = False

                one_hot_labels = None
                if self.apply_region_growing:
                    if is_pixel_mask:
                        one_hot_labels = masks
                    else:
                        one_hot_labels = self._to_one_hot(masks)

                outputs = self.model(images, labels=one_hot_labels, apply_region_growing=self.apply_region_growing)
                logits = outputs['logits']
                segmentation_maps = outputs['segmentation_maps']

                # Class prediction loss
                cls_loss = self.cls_criterion(logits, batch['class_label'].to(self.device))
                
                # Segmentation loss
                if is_pixel_mask:
                    seg_loss = self._calculate_pixel_supervision_loss(segmentation_maps, masks)
                else:
                    seg_loss = self._calculate_weak_supervision_loss(segmentation_maps, masks)
                
                loss = cls_loss + self.seg_loss_weight * seg_loss

                total_loss += loss.item()
                count += 1

        return total_loss / count if count > 0 else 0.0

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

    def _calculate_weak_supervision_loss(self, segmentation_maps, labels):
        batch_size = segmentation_maps.size(0)
        loss = 0
        for i in range(batch_size):
            seg_map = segmentation_maps[i]
            label_idx = labels[i].item()
            target_activation = torch.sigmoid(seg_map[label_idx])
            binary_mask = (target_activation > 0.5).float()
            intersection = (target_activation * binary_mask).sum()
            union = target_activation.sum() + binary_mask.sum()
            dice_score = (2 * intersection + 1e-6) / (union + 1e-6)
            loss += (1 - dice_score)
        return loss / batch_size

    def _to_one_hot(self, labels):
        batch_size = labels.size(0)
        num_classes = self.model.num_classes if hasattr(self.model, 'num_classes') else self.config['model']['num_classes']
        one_hot = torch.zeros(batch_size, num_classes, device=self.device)
        one_hot[torch.arange(batch_size), labels] = 1.0
        return one_hot

    def _save_checkpoint(self, epoch, loss):
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

        os.makedirs(self.log_dir / "plots", exist_ok=True)
        image.save(self.log_dir / "plots" / "loss_plot.png")
        logging.info(f"Loss plot saved to {self.log_dir / 'plots' / 'loss_plot.png'}")