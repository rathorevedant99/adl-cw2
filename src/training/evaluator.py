import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import random
import logging
from PIL import Image, ImageDraw
import torchvision.transforms as T

class Evaluator:
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
        self.eval_loader = DataLoader(
            dataset,
            batch_size=config['evaluation']['batch_size'],
            shuffle=False,
            num_workers=config['evaluation']['num_workers']
        )
        
        # Setup visualization directory
        self.viz_dir = Path(config.get('viz_dir', 'experiments/visualizations'))
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
    def evaluate(self):
        self.model.eval()
        metrics = {
            'accuracy': 0,
            'mean_iou': 0,
            'pixel_accuracy': 0
        }
        
        all_cls_preds = []
        all_cls_labels = []
        all_seg_preds = []
        all_seg_labels = []
        
        logging.info('Starting model evaluation...')
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.eval_loader):
                # Move data to device
                images = batch['image'].to(self.device)
                seg_labels = batch['mask'].to(self.device)
                
                # Get class labels from segmentation masks (assuming each image belongs to one class)
                # This is a simplification - you might need to adjust based on your actual data
                cls_labels = torch.zeros(seg_labels.size(0), dtype=torch.long, device=self.device)
                for i in range(seg_labels.size(0)):
                    # Get the most common class in the segmentation mask
                    unique, counts = torch.unique(seg_labels[i], return_counts=True)
                    cls_labels[i] = unique[torch.argmax(counts)]
                
                # Forward pass
                outputs = self.model(images)
                logits = outputs['logits']
                segmentation_maps = outputs['segmentation_maps']
                
                # Get predictions
                cls_preds = torch.argmax(logits, dim=1)
                seg_preds = torch.argmax(segmentation_maps, dim=1)
                
                # Calculate metrics
                batch_metrics = self._calculate_metrics(
                    cls_preds,
                    seg_preds,
                    cls_labels,
                    seg_labels
                )
                
                # Update metrics
                for k, v in batch_metrics.items():
                    metrics[k] += v
                
                # Store predictions and labels
                all_cls_preds.extend(cls_preds.cpu().numpy())
                all_cls_labels.extend(cls_labels.cpu().numpy())
                all_seg_preds.extend(seg_preds.cpu().numpy())
                all_seg_labels.extend(seg_labels.cpu().numpy())
                
                # Log progress
                if batch_idx % 10 == 0:
                    logging.info(f'Evaluated batch {batch_idx}/{len(self.eval_loader)}')
        
        # Calculate average metrics
        num_batches = len(self.eval_loader)
        for k in metrics:
            metrics[k] /= num_batches
        
        # Calculate confusion matrix using PyTorch
        conf_matrix = self._calculate_confusion_matrix(
            torch.tensor(all_cls_preds),
            torch.tensor(all_cls_labels),
            num_classes=self.config['model']['num_classes']
        )
        
        # Log metrics
        logging.info("\nEvaluation Results:")
        logging.info(f"Classification Accuracy: {metrics['accuracy']:.4f}")
        logging.info(f"Mean IoU: {metrics['mean_iou']:.4f}")
        logging.info(f"Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
        logging.info("\nConfusion Matrix:")
        logging.info(f"\n{conf_matrix}")
        
        # Visualize CAMs for sample images
        self._visualize_cams()
        
        return metrics
    
    def _calculate_metrics(self, cls_preds, seg_preds, cls_labels, seg_labels):
        """Calculate evaluation metrics"""
        batch_size = cls_preds.size(0)
        metrics = {
            'accuracy': 0,
            'mean_iou': 0,
            'pixel_accuracy': 0
        }
        
        for i in range(batch_size):
            # Classification accuracy - compare predicted class with ground truth class
            metrics['accuracy'] += (cls_preds[i] == cls_labels[i]).float().item()
            
            # Segmentation metrics
            pred_mask = seg_preds[i]
            true_mask = seg_labels[i]
            
            # Calculate IoU for each class
            ious = []
            for class_idx in range(self.config['model']['num_classes']):
                pred_class = (pred_mask == class_idx)
                true_class = (true_mask == class_idx)
                
                intersection = (pred_class & true_class).sum().float()
                union = (pred_class | true_class).sum().float()
                
                iou = (intersection + 1e-8) / (union + 1e-8)
                ious.append(iou.item())
            
            metrics['mean_iou'] += np.mean(ious)
            metrics['pixel_accuracy'] += (pred_mask == true_mask).float().mean().item()
        
        return metrics
    
    def _calculate_confusion_matrix(self, preds, labels, num_classes):
        """Calculate confusion matrix using PyTorch"""
        conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
        
        for t, p in zip(labels, preds):
            conf_matrix[t, p] += 1
            
        return conf_matrix
    
    def _create_heatmap(self, cam, size):
        """Convert CAM to PIL heatmap image"""
        # Normalize CAM
        cam = np.maximum(cam, 0)  # ReLU
        cam = cam / (cam.max() + 1e-8)  # Normalize
        
        # Convert to PIL Image
        cam_pil = Image.fromarray((cam * 255).astype(np.uint8))
        cam_pil = cam_pil.resize(size, Image.Resampling.LANCZOS)
        
        # Convert to RGB with jet colormap
        cam_rgb = Image.new('RGB', size)
        for x in range(size[0]):
            for y in range(size[1]):
                value = cam_pil.getpixel((x, y)) / 255.0
                # Simple jet colormap approximation
                r = min(255, max(0, int(255 * (1.5 - abs(4 * value - 3)))))
                g = min(255, max(0, int(255 * (1.5 - abs(4 * value - 2)))))
                b = min(255, max(0, int(255 * (1.5 - abs(4 * value - 1)))))
                cam_rgb.putpixel((x, y), (r, g, b))
        
        return cam_rgb
    
    def _create_overlay(self, original_img, cam_img, alpha=0.5):
        """Create overlay of CAM on original image"""
        overlay = Image.blend(original_img.convert('RGB'), cam_img, alpha)
        return overlay
    
    def _visualize_cams(self, num_samples=5):
        """Visualize and save CAMs for sample images using PIL"""
        logging.info(f"\nGenerating CAM visualizations for {num_samples} sample images...")
        
        # Get a random subset of the dataset
        indices = random.sample(range(len(self.dataset)), min(num_samples, len(self.dataset)))
        
        # Create visualization for each sample
        for idx in indices:
            # Get the sample
            sample = self.dataset[idx]
            image = sample['image'].unsqueeze(0).to(self.device)  # Add batch dimension
            true_label = sample['mask']
            image_name = sample['image_name']
            
            # Get model outputs
            with torch.no_grad():
                outputs = self.model(image)
                logits = outputs['logits']
                segmentation_maps = outputs['segmentation_maps']
                
                # Get the predicted class
                pred_class = torch.argmax(logits, dim=1).item()
                
                # Get the CAM for both true and predicted classes
                true_cam = segmentation_maps[0, true_label].cpu().numpy()
                pred_cam = segmentation_maps[0, pred_class].cpu().numpy()
                
                # Convert image to PIL
                img_np = image[0].cpu().numpy().transpose(1, 2, 0)
                img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
                img_np = img_np.astype(np.uint8)
                original_img = Image.fromarray(img_np)
                
                # Get image size
                width, height = original_img.size
                
                # Create heatmaps
                true_heatmap = self._create_heatmap(true_cam, (width, height))
                pred_heatmap = self._create_heatmap(pred_cam, (width, height))
                
                # Create overlays
                true_overlay = self._create_overlay(original_img, true_heatmap)
                pred_overlay = self._create_overlay(original_img, pred_heatmap)
                
                # Create final visualization
                final_img = Image.new('RGB', (width * 5, height))
                final_img.paste(original_img, (0, 0))
                final_img.paste(true_heatmap, (width, 0))
                final_img.paste(true_overlay, (width * 2, 0))
                final_img.paste(pred_heatmap, (width * 3, 0))
                final_img.paste(pred_overlay, (width * 4, 0))
                
                # Add text labels
                draw = ImageDraw.Draw(final_img)
                draw.text((10, 10), f"Original: {image_name}", fill='white')
                draw.text((width + 10, 10), f"True Class {true_label} CAM", fill='white')
                draw.text((width * 2 + 10, 10), f"True Class Overlay", fill='white')
                draw.text((width * 3 + 10, 10), f"Pred Class {pred_class} CAM", fill='white')
                draw.text((width * 4 + 10, 10), f"Pred Class Overlay", fill='white')
                
                # Save the visualization
                final_img.save(self.viz_dir / f'cam_{image_name}.png')
                logging.info(f"Saved CAM visualization for {image_name} (True: {true_label}, Pred: {pred_class})")
        
        logging.info(f"All CAM visualizations saved to {self.viz_dir}") 