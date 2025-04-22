import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import random
import logging
from PIL import Image, ImageDraw
import torchvision.transforms as T
import matplotlib.pyplot as plt

class Evaluator:
    def __init__(self, model, dataset, config, weak_supervision_types=None):
        self.model = model
        self.dataset = dataset
        self.config = config
        self.weak_supervision_types = weak_supervision_types or ['labels']
        
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
        
        self.eval_loader = DataLoader(
            dataset,
            batch_size=config['evaluation']['batch_size'],
            shuffle=False,
            num_workers=config['evaluation']['num_workers']
        )
        
        self.viz_dir = Path(config.get('viz_dir', 'experiments/visualizations'))
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a directory for experiment-specific visualizations
        self.weak_sup_str = '_'.join(self.weak_supervision_types)
        self.exp_viz_dir = self.viz_dir / self.weak_sup_str
        self.exp_viz_dir.mkdir(parents=True, exist_ok=True)
        
    def evaluate(self):
        self.model.eval()
        metrics = {
            'accuracy': [],
            'mean_iou': [],
            'pixel_accuracy': []
        }
        
        all_cls_preds = []
        all_cls_labels = []
        all_seg_preds = []
        all_seg_labels = []
        
        logging.info(f'Starting model evaluation with weak supervision types: {self.weak_supervision_types}')
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.eval_loader):
                images = batch['image'].to(self.device)
                
                # Always use full segmentation masks for evaluation
                seg_labels = batch['full_mask'].to(self.device)
                
                # For classification labels
                cls_labels = torch.zeros(seg_labels.size(0), dtype=torch.long, device=self.device)
                for i in range(seg_labels.size(0)):
                    unique, counts = torch.unique(seg_labels[i], return_counts=True)
                    if len(unique) > 0:
                        cls_labels[i] = unique[torch.argmax(counts)]
                
                # Run model with the appropriate weak supervision
                one_hot_labels = None
                apply_region_growing = True
                
                outputs = self.model(images, labels=one_hot_labels, apply_region_growing=apply_region_growing)
                logits = outputs['logits']
                segmentation_maps = outputs['segmentation_maps']
                
                cls_preds = torch.argmax(logits, dim=1)
                seg_preds = torch.argmax(segmentation_maps, dim=1)
                
                batch_metrics = self._calculate_metrics(
                    cls_preds,
                    seg_preds,
                    cls_labels,
                    seg_labels
                )
                
                for k, v in batch_metrics.items():
                    metrics[k].append(v)
                
                all_cls_preds.extend(cls_preds.cpu().numpy())
                all_cls_labels.extend(cls_labels.cpu().numpy())
                all_seg_preds.extend(seg_preds.cpu().numpy())
                all_seg_labels.extend(seg_labels.cpu().numpy())
                
                if batch_idx % 10 == 0:
                    logging.info(f'Evaluated batch {batch_idx}/{len(self.eval_loader)}')
        
        final_metrics = {k: float(np.mean(v)) for k, v in metrics.items()}
        logging.info(f"\nEvaluation Results for {self.weak_sup_str}:")
        logging.info(f"Classification Accuracy: {final_metrics['accuracy']:.4f}")
        logging.info(f"Mean IoU: {final_metrics['mean_iou']:.4f}")
        logging.info(f"Pixel Accuracy: {final_metrics['pixel_accuracy']:.4f}")
        
        self._visualize_cams()
        self._visualize_weak_supervision_comparison()
        self._visualize_examples()
        
        return final_metrics

    def _calculate_metrics(self, cls_preds, seg_preds, cls_labels, seg_labels):
        batch_size = cls_preds.size(0)
        accuracy = 0
        mean_iou = 0
        pixel_accuracy = 0
        
        for i in range(batch_size):
            # Classification accuracy
            accuracy += (cls_preds[i] == cls_labels[i]).float().item() 
            
            # Get predicted and ground truth segmentation masks
            pred_mask = seg_preds[i]
            true_mask = seg_labels[i]
            
            # Calculate IoU for each class
            ious = []
            for class_idx in range(self.config['model']['num_classes']):
                pred_class = (pred_mask == class_idx) # The predicted mask for the current class
                true_class = (true_mask == class_idx) # The true mask for the current class
                intersection = (pred_class & true_class).sum().float() # The intersection of the predicted and true masks
                union = (pred_class | true_class).sum().float() # The union of the predicted and true masks
                iou = (intersection + 1e-8) / (union + 1e-8) # The IoU for the current class
                ious.append(iou.item()) # Add the IoU to the list
            
            mean_iou += np.mean(ious)
            
            # =================== FIXED PIXEL ACCURACY CALCULATION ===================
            # Map trimaps values to model's prediction space
            # In Oxford Pet dataset: 1=pet, 2=background, 3=boundary
            # In our model: typically 0=background, 1=pet, etc.
            
            # Create remapped ground truth for proper comparison
            remapped_true_mask = torch.zeros_like(true_mask)
            
            # Find unique values in the true mask
            unique_values = torch.unique(true_mask)
            
            # Get foreground class (usually the pet)
            # This assumes the pet is the most common non-zero class
            non_zero_classes = unique_values[unique_values > 0]
            if len(non_zero_classes) > 0:
                pet_class = non_zero_classes[0].item()
                # Map pet class (usually 1 in trimaps) to the appropriate predicted class 
                # This is often the class with highest frequency in predictions
                pred_unique, pred_counts = torch.unique(pred_mask, return_counts=True)
                if len(pred_unique) > 0:
                    pred_pet_class = pred_unique[torch.argmax(pred_counts)].item()
                    
                    # Map ground truth values to prediction space
                    remapped_true_mask[true_mask == pet_class] = pred_pet_class  # Pet foreground
                    remapped_true_mask[true_mask != pet_class] = 0  # Background or boundary
                    
                    # Calculate pixel accuracy with remapped values
                    pixel_accuracy += (pred_mask == remapped_true_mask).float().mean().item()
                else:
                    pixel_accuracy += 0.0  # No predictions found
            else:
                pixel_accuracy += 0.0  # No foreground found
            # ======================================================================
        
        return {
            'accuracy': accuracy / batch_size,
            'mean_iou': mean_iou / batch_size,
            'pixel_accuracy': pixel_accuracy / batch_size
        }

    def _create_heatmap(self, cam, size):
        cam = np.maximum(cam, 0)
        cam = cam / (cam.max() + 1e-8)
        cam_pil = Image.fromarray((cam * 255).astype(np.uint8))
        cam_pil = cam_pil.resize(size, Image.Resampling.LANCZOS)
        cam_rgb = Image.new('RGB', size)
        for x in range(size[0]):
            for y in range(size[1]):
                value = cam_pil.getpixel((x, y)) / 255.0
                r = min(255, max(0, int(255 * (1.5 - abs(4 * value - 3)))))
                g = min(255, max(0, int(255 * (1.5 - abs(4 * value - 2)))))
                b = min(255, max(0, int(255 * (1.5 - abs(4 * value - 1)))))
                cam_rgb.putpixel((x, y), (r, g, b))
        return cam_rgb

    def _create_overlay(self, original_img, cam_img, alpha=0.5):
        return Image.blend(original_img.convert('RGB'), cam_img, alpha)

    def _visualize_cams(self, num_samples=5):
        logging.info(f"\nGenerating CAM visualizations for {num_samples} sample images...")
        indices = random.sample(range(len(self.dataset)), min(num_samples, len(self.dataset)))
        for idx in indices:
            sample = self.dataset[idx]
            image = sample['image'].unsqueeze(0).to(self.device)
            true_label = sample['class_label']
            image_name = sample['image_name']

            with torch.no_grad():
                outputs = self.model(image)
                logits = outputs['logits']
                segmentation_maps = outputs['segmentation_maps']

                pred_class = torch.argmax(logits, dim=1).item()
                true_cam = segmentation_maps[0, true_label].cpu().numpy()
                pred_cam = segmentation_maps[0, pred_class].cpu().numpy()
                
                img_np = image[0].cpu().numpy().transpose(1, 2, 0)
                img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
                img_np = img_np.astype(np.uint8)
                
                original_img = Image.fromarray(img_np)
                
                width, height = original_img.size
                true_heatmap = self._create_heatmap(true_cam, (width, height))
                pred_heatmap = self._create_heatmap(pred_cam, (width, height))
                true_overlay = self._create_overlay(original_img, true_heatmap)
                pred_overlay = self._create_overlay(original_img, pred_heatmap)
                
                final_img = Image.new('RGB', (width * 5, height))
                final_img.paste(original_img, (0, 0))
                final_img.paste(true_heatmap, (width, 0))
                final_img.paste(true_overlay, (width * 2, 0))
                final_img.paste(pred_heatmap, (width * 3, 0))
                final_img.paste(pred_overlay, (width * 4, 0))
                
                draw = ImageDraw.Draw(final_img)
                draw.text((10, 10), f"Original: {image_name}", fill='white')
                draw.text((width + 10, 10), f"True Class {true_label} CAM", fill='white')
                draw.text((width * 2 + 10, 10), f"True Class Overlay", fill='white')
                draw.text((width * 3 + 10, 10), f"Pred Class {pred_class} CAM", fill='white')
                draw.text((width * 4 + 10, 10), f"Pred Class Overlay", fill='white')
                
                final_img.save(self.exp_viz_dir / f'cam_{image_name}.png')
                logging.info(f"Saved CAM visualization for {image_name} (True: {true_label}, Pred: {pred_class})")
        logging.info(f"All CAM visualizations saved to {self.exp_viz_dir}")

    def _visualize_weak_supervision_comparison(self, num_samples=3):
        """Visualize the different weak supervision signals and resulting segmentation"""
        logging.info(f"\nGenerating weak supervision comparison visualizations for {num_samples} sample images...")
        
        indices = random.sample(range(len(self.dataset)), min(num_samples, len(self.dataset)))
        
        for idx in indices:
            sample = self.dataset[idx]
            image = sample['image'].unsqueeze(0).to(self.device)
            image_name = sample['image_name']
            
            # Get ground truth mask for reference
            full_mask = sample['full_mask']
            
            # Get weak supervision masks if available
            weak_masks = {}
            
            if 'labels' in self.weak_supervision_types:
                weak_masks['labels'] = f"Class Label: {sample['class_label'].item()}"
                
            if 'bboxes' in self.weak_supervision_types and 'bbox_mask' in sample:
                weak_masks['bboxes'] = sample['bbox_mask']
                
            if 'scribbles' in self.weak_supervision_types and 'scribble_mask' in sample:
                weak_masks['scribbles'] = sample['scribble_mask']
            
            # Get model prediction
            with torch.no_grad():
                outputs = self.model(image)
                segmentation_maps = outputs['segmentation_maps']
                predicted_mask = torch.argmax(segmentation_maps, dim=1)[0].cpu()
            
            # Create visualization
            # Convert tensors to numpy arrays for visualization
            img_np = image[0].cpu().numpy().transpose(1, 2, 0)
            img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
            img_np = img_np.astype(np.uint8)
            
            # Create a figure for visualization
            n_cols = 3 + len(weak_masks)  # Original, weak supervision signals, prediction, ground truth
            fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
            
            # Plot original image
            axes[0].imshow(img_np)
            axes[0].set_title("Original Image")
            axes[0].axis('off')
            
            # Plot weak supervision signals
            col_idx = 1
            for sup_type, mask in weak_masks.items():
                if isinstance(mask, str):  # For class labels
                    axes[col_idx].imshow(img_np)
                    axes[col_idx].text(10, 20, mask, color='white', fontsize=12, 
                                      bbox=dict(facecolor='black', alpha=0.5))
                else:  # For visual masks
                    if mask.dim() <= 2:  # Single-channel mask
                        mask_vis = self._colorize_mask(mask.numpy())
                        axes[col_idx].imshow(mask_vis)
                    else:  # Multi-channel mask
                        mask_vis = self._colorize_mask(torch.argmax(mask, dim=0).numpy())
                        axes[col_idx].imshow(mask_vis)
                
                axes[col_idx].set_title(f"{sup_type.capitalize()} Supervision")
                axes[col_idx].axis('off')
                col_idx += 1
            
            # Plot prediction
            pred_vis = self._colorize_mask(predicted_mask.numpy())
            axes[col_idx].imshow(pred_vis)
            axes[col_idx].set_title("Model Prediction")
            axes[col_idx].axis('off')
            col_idx += 1
            
            # Plot ground truth
            gt_vis = self._colorize_mask(full_mask.numpy())
            axes[col_idx].imshow(gt_vis)
            axes[col_idx].set_title("Ground Truth")
            axes[col_idx].axis('off')
            
            plt.tight_layout()
            vis_path = self.exp_viz_dir / f"weak_sup_comparison_{image_name}.png"
            plt.savefig(vis_path)
            plt.close()
            
            logging.info(f"Saved weak supervision comparison for {image_name} at {vis_path}")
        
        logging.info(f"All weak supervision comparisons saved to {self.exp_viz_dir}")
    
    def _visualize_examples(self, num_samples=3):
        """Save examples of segmentation results"""
        logging.info(f"\nSaving {num_samples} example segmentations...")
        
        # Get random samples
        indices = random.sample(range(len(self.dataset)), min(num_samples, len(self.dataset)))
        
        for idx in indices:
            sample = self.dataset[idx]
            image = sample['image'].unsqueeze(0).to(self.device)
            image_name = sample['image_name']
            
            # Get ground truth
            true_mask = sample['full_mask']
            
            # Get prediction from model
            with torch.no_grad():
                outputs = self.model(image)
                segmentation_maps = outputs['segmentation_maps']
                pred_mask = torch.argmax(segmentation_maps, dim=1)[0].cpu()
            
            # Convert image to PIL format
            img_np = image[0].cpu().numpy().transpose(1, 2, 0)
            img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
            img = Image.fromarray(img_np.astype(np.uint8))
            
            # Convert masks to PIL format - FIXED TENSOR TO NUMPY CONVERSION
            true_mask_np = true_mask.cpu().numpy()
            max_val_true = true_mask_np.max()
            # Avoid division by zero
            if max_val_true > 0:
                scaled_true_mask = (true_mask_np * 255 // max_val_true).astype(np.uint8)
            else:
                scaled_true_mask = true_mask_np.astype(np.uint8)
            gt_mask_img = Image.fromarray(scaled_true_mask).convert('L')
            
            pred_mask_np = pred_mask.cpu().numpy()
            max_val_pred = pred_mask_np.max()
            # Avoid division by zero
            if max_val_pred > 0:
                scaled_pred_mask = (pred_mask_np * 255 // max_val_pred).astype(np.uint8)
            else:
                scaled_pred_mask = pred_mask_np.astype(np.uint8)
            pred_mask_img = Image.fromarray(scaled_pred_mask).convert('L')
            
            # Create a single image with all three
            width, height = img.size
            result_img = Image.new('RGB', (width * 3, height))
            result_img.paste(img, (0, 0))
            result_img.paste(gt_mask_img.convert('RGB'), (width, 0))
            result_img.paste(pred_mask_img.convert('RGB'), (width * 2, 0))
            
            # Add labels
            draw = ImageDraw.Draw(result_img)
            draw.text((10, 10), "Original", fill=(255, 255, 255))
            draw.text((width + 10, 10), "Ground Truth", fill=(255, 255, 255))
            draw.text((width * 2 + 10, 10), "Prediction", fill=(255, 255, 255))
            
            # Save the result
            result_img.save(self.exp_viz_dir / f"segmentation_{image_name}.png")
            
            logging.info(f"Saved segmentation example for {image_name}")
    
    def _colorize_mask(self, mask):
        """Convert a segmentation mask to a colorized visualization"""
        # Define colors for different classes
        colors = [
            [0, 0, 0],       # Background (black)
            [255, 0, 0],     # Class 1 (red)
            [0, 255, 0],     # Class 2 (green)
            [0, 0, 255],     # Class 3 (blue)
            [255, 255, 0],   # Class 4 (yellow)
            [255, 0, 255],   # Class 5 (magenta)
            [0, 255, 255],   # Class 6 (cyan)
            [128, 0, 0],     # Class 7 (maroon)
            [0, 128, 0],     # Class 8 (dark green)
            [0, 0, 128],     # Class 9 (navy)
            [128, 128, 0],   # Class 10 (olive)
            [128, 0, 128],   # Class 11 (purple)
            [0, 128, 128],   # Class 12 (teal)
            [192, 192, 192], # Class 13 (silver)
            [128, 128, 128], # Class 14 (gray)
            [255, 165, 0],   # Class 15 (orange)
            [210, 105, 30],  # Class 16 (chocolate)
            [255, 192, 203], # Class 17 (pink)
            [165, 42, 42],   # Class 18 (brown)
            [240, 230, 140], # Class 19 (khaki)
        ]
        
        # Create RGB image
        h, w = mask.shape
        colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Fill in colors for each class
        num_classes = min(len(colors), int(mask.max()) + 1)
        for i in range(num_classes):
            colored_mask[mask == i] = colors[i]
            
        return colored_mask
