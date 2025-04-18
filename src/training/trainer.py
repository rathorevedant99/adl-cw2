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
        self.apply_region_growing = config['training'].get('apply_region_growing', True)

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

                # Debug labels
                if epoch == 0 and batch_idx < 2:
                    logging.info(f"Label shape: {labels.shape}")
                    logging.info(f"Label dtype: {labels.dtype}")
                    logging.info(f"Label values: {labels}")
                    if len(labels.shape) == 1:
                        logging.info("CONFIRMATION: Labels are scalar values (one per image)")
                    else:
                        logging.info(f"Labels have shape {labels.shape} - they appear to be masks, not scalars")

                # Convert to one-hot for region growing
                one_hot_labels = self._to_one_hot(labels) if self.apply_region_growing else None

                # Forward pass with region growing
                outputs = self.model(images, labels=one_hot_labels, apply_region_growing=self.apply_region_growing)
                logits = outputs['logits']
                segmentation_maps = outputs['segmentation_maps']

                # Classification loss
                cls_loss = self.cls_criterion(logits, labels)

                # Weak supervision loss
                seg_loss = self._calculate_weak_supervision_loss(segmentation_maps, labels)

                # Total loss
                loss = cls_loss + self.seg_loss_weight * seg_loss

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Track loss
                batch_loss = loss.item()
                epoch_loss += batch_loss
                epoch_batch_losses.append(batch_loss)
                self.batch_losses.append(batch_loss)

                if batch_idx % 10 == 0:
                    logging.info(f'Batch {batch_idx}/{len(self.train_loader)}, Loss: {batch_loss:.4f}')

            # Epoch summary
            avg_epoch_loss = epoch_loss / len(self.train_loader)
            self.epoch_losses.append(avg_epoch_loss)
            logging.info(f'Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}')

            # Save checkpoint
            if (epoch + 1) % self.save_interval == 0:
                self._save_checkpoint(epoch + 1, avg_epoch_loss)

        # Plot loss
        self._plot_loss()

    def _plot_loss(self):
        """Plot training loss over epochs using PIL/Pillow"""
        
        # Create a blank white image
        width, height = 1000, 500
        margin = 50
        plot_width = width - 2 * margin
        plot_height = height - 2 * margin
        
        image = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(image)
        
        # Draw frame and axes
        draw.rectangle([(margin, margin), (width - margin, height - margin)], outline='black')
        
        # Draw axes labels
        try:
            # Try to load a font - use default if not available
            font = ImageFont.truetype("arial.ttf", 12)
        except IOError:
            font = ImageFont.load_default()
        
        draw.text((width // 2, height - margin // 2), "Epochs", fill='black', anchor="mm", font=font)
        draw.text((margin // 2, height // 2), "Loss", fill='black', anchor="mm", font=font, angle=90)
        draw.text((width // 2, margin // 2), "Training Loss", fill='black', anchor="mm", font=font)
        
        # Normalize loss values for plotting
        if self.epoch_losses:
            losses = np.array(self.epoch_losses)
            min_loss = np.min(losses)
            max_loss = np.max(losses)
            
            # Add a small buffer to the range to avoid plotting on the edges
            loss_range = max_loss - min_loss
            if loss_range == 0:  # Handle case where all losses are the same
                loss_range = max_loss * 0.1 or 0.1
                
            min_loss -= loss_range * 0.05
            max_loss += loss_range * 0.05
            
            # Calculate x and y coordinates for each point
            num_points = len(losses)
            x_step = plot_width / (num_points - 1) if num_points > 1 else plot_width
            
            points = []
            for i, loss in enumerate(losses):
                x = margin + i * x_step
                # Normalize and invert y (PIL y-axis increases downward)
                y = margin + plot_height - plot_height * (loss - min_loss) / (max_loss - min_loss)
                points.append((x, y))
                
            # Draw lines connecting points
            if len(points) > 1:
                for i in range(len(points) - 1):
                    draw.line([points[i], points[i+1]], fill='blue', width=2)
            
            # Draw points
            for x, y in points:
                draw.ellipse((x-3, y-3, x+3, y+3), fill='blue')
                
            # Draw y-axis labels (loss values)
            num_y_ticks = 5
            for i in range(num_y_ticks + 1):
                y_pos = margin + plot_height - i * plot_height / num_y_ticks
                tick_value = min_loss + i * (max_loss - min_loss) / num_y_ticks
                draw.line([(margin - 5, y_pos), (margin, y_pos)], fill='black')
                draw.text((margin - 10, y_pos), f"{tick_value:.4f}", fill='black', anchor="rm", font=font)
                
            # Draw x-axis labels (epochs)
            num_x_ticks = min(num_points, 10)  # Limit number of ticks to avoid overcrowding
            for i in range(num_x_ticks + 1):
                epoch = i * (num_points - 1) // num_x_ticks if num_points > 1 else 0
                x_pos = margin + epoch * x_step
                draw.line([(x_pos, height - margin), (x_pos, height - margin + 5)], fill='black')
                draw.text((x_pos, height - margin + 10), str(epoch), fill='black', anchor="mt", font=font)
                
            # Draw legend
            legend_x = width - margin - 100
            legend_y = margin + 20
            draw.rectangle([(legend_x, legend_y), (legend_x + 80, legend_y + 20)], outline='black')
            draw.line([(legend_x + 10, legend_y + 10), (legend_x + 30, legend_y + 10)], fill='blue', width=2)
            draw.ellipse((legend_x + 20 - 3, legend_y + 10 - 3, legend_x + 20 + 3, legend_y + 10 + 3), fill='blue')
            draw.text((legend_x + 40, legend_y + 10), "Epoch Loss", fill='black', anchor="lm", font=font)
            
        # Make sure directory exists
        os.makedirs("experiments/plots", exist_ok=True)
        
        # Save the image
        image.save("experiments/plots/loss_plot.png")
        print("Loss plot saved to experiments/plots/loss_plot.png")

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
            'config': self.config
        }

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        logging.info(f'Saved checkpoint to {checkpoint_path}')
