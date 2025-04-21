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
    def __init__(self, model, dataset, config, weak_supervision_types=None):
        self.model = model
        self.dataset = dataset
        self.config = config
        self.weak_supervision_types = weak_supervision_types or ['labels']
        self.method = config['model'].get('method', 'WS').upper()
        
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
            num_workers=config['evaluation']['num_workers'],
            pin_memory=config['training'].get('pin_memory', False)
        )
        
        self.viz_dir = Path(config.get('viz_dir', 'experiments/visualizations'))
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a directory for experiment-specific visualizations
        self.weak_sup_str = '_'.join(self.weak_supervision_types)
        self.exp_viz_dir = self.viz_dir / self.weak_sup_str
        self.exp_viz_dir.mkdir(parents=True, exist_ok=True)
        
    def evaluate(self):
        self.model.eval()
        metrics = {'accuracy': [], 'mean_iou': [], 'pixel_accuracy': []}
        
        logging.info(f'Starting model evaluation with method: {self.method}...')
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.eval_loader):
                # Unpack batch data based on method
                if self.method == 'WS':
                    if isinstance(batch, tuple) and len(batch) == 2:
                        weak_batch, fs_batch = batch
                        images = weak_batch['image'].to(self.device)
                        cls_labels = weak_batch['mask'].to(self.device)  # scalar class idx
                        seg_labels = fs_batch['mask'].to(self.device)    # H×W mask
                    else:
                        # Handle case when batch isn't a tuple
                        images = batch['image'].to(self.device)
                        if 'class_label' in batch:
                            cls_labels = batch['class_label'].to(self.device)
                        else:
                            cls_labels = batch['mask'].to(self.device)
                        seg_labels = batch['mask'].to(self.device)
                else:
                    # Fully supervised mode
                    images = batch['image'].to(self.device)
                    seg_labels = batch['mask'].to(self.device)
                    cls_labels = None
                
                # Forward pass
                outputs = self.model(images)
                
                if self.method == 'WS':
                    # Process for weakly supervised method
                    cams = outputs['segmentation_maps']           # [B, C, H, W]
                    idx = torch.arange(cams.size(0), device=self.device)
                    pet_cams = cams[idx, cls_labels]             # [B, H, W]

                    # Threshold to binary mask
                    prob_maps = torch.sigmoid(pet_cams)
                    seg_preds = (prob_maps > 0.5).long()

                    # Compute binary IoU + pixel accuracy
                    mean_iou, pixel_acc = self._calc_binary_metrics(seg_preds, seg_labels)

                    # Classification accuracy
                    logits = outputs['logits']
                    cls_preds = logits.argmax(dim=1)
                    acc = (cls_preds == cls_labels).float().mean().item()
                    metrics['accuracy'].append(acc)
                else:
                    # Process for fully supervised method
                    seg_maps = outputs['segmentation_maps']
                    seg_preds = seg_maps.argmax(dim=1)          # [B, H, W]
                    mean_iou, pixel_acc = self._calc_seg_metrics(seg_preds, seg_labels)
                
                metrics['mean_iou'].append(mean_iou)
                metrics['pixel_accuracy'].append(pixel_acc)
                
                if batch_idx % 10 == 0:
                    logging.info(f'Evaluated batch {batch_idx}/{len(self.eval_loader)}')
        
        # Aggregate metrics
        final_metrics = {k: float(np.mean(v)) for k, v in metrics.items()}
        
        logging.info(f"\nEvaluation Results for {self.weak_sup_str}:")
        if self.method == 'WS':
            logging.info(f"Classification Accuracy: {final_metrics['accuracy']:.4f}")
        logging.info(f"Mean IoU: {final_metrics['mean_iou']:.4f}")
        logging.info(f"Pixel Accuracy: {final_metrics['pixel_accuracy']:.4f}")
        
        # Visualize results
        if self.method == 'WS':
            self._visualize_cams()
        self._visualize_examples()
        
        return final_metrics

    def _calc_seg_metrics(self, preds, labels):
        # Compute mean IoU and pixel accuracy for a batch
        batch_size = preds.size(0)
        total_iou, total_acc = 0.0, 0.0
        num_classes = self.config['model']['num_classes']
        
        for i in range(batch_size):
            pred_mask = preds[i]
            true_mask = labels[i]
            # pixel accuracy
            total_acc += (pred_mask == true_mask).float().mean().item()
            # iou per class
            ious = []
            for c in range(num_classes):
                p = (pred_mask == c)
                t = (true_mask == c)
                union = (p | t).sum().float()
                if union > 0:
                    intersection = (p & t).sum().float()
                    ious.append((intersection / (union + 1e-8)).item())
            if ious:
                total_iou += np.mean(ious)
        return total_iou / batch_size, total_acc / batch_size

    def _calc_binary_metrics(self, preds, true_masks):
        """
        preds, true_masks: LongTensors of shape [B, H, W] with values 0 or 1.
        Returns (mean_iou, pixel_acc).
        """
        batch_size = preds.size(0)
        total_iou, total_acc = 0.0, 0.0

        for i in range(batch_size):
            p = preds[i].bool()
            t = true_masks[i].bool()

            inter = (p & t).sum().float()
            uni = (p | t).sum().float() + 1e-8
            iou = (inter / uni).item()
            acc = (p == t).float().mean().item()

            total_iou += iou
            total_acc += acc

        return total_iou / batch_size, total_acc / batch_size

    def _create_heatmap(self, cam, size):
        cam = np.clip(cam, 0, None)
        cam = cam / (cam.max() + 1e-8)
        cam_pil = Image.fromarray((cam * 255).astype(np.uint8)).resize(size, Image.Resampling.LANCZOS)
        cam_rgb = Image.new('RGB', size)
        for x in range(size[0]):
            for y in range(size[1]):
                v = cam_pil.getpixel((x,y)) / 255.0
                r = int(255 * max(0, min(1, 1.5 - abs(4*v-3))))
                g = int(255 * max(0, min(1, 1.5 - abs(4*v-2))))
                b = int(255 * max(0, min(1, 1.5 - abs(4*v-1))))
                cam_rgb.putpixel((x,y),(r,g,b))
        return cam_rgb

    def _visualize_cams(self, num_samples=5):
        logging.info(f"\nGenerating CAM visualizations for {num_samples} sample images...")
        indices = random.sample(range(len(self.dataset)), min(num_samples, len(self.dataset)))
        for idx in indices:
            sample = self.dataset[idx]
            image = sample['image'].unsqueeze(0).to(self.device)
            
            # Handle different mask structures
            if 'class_label' in sample:
                true_label = sample['class_label']
            else:
                true_label = sample['mask']
                if true_label.dim() > 0:
                    # If mask is a 2D segmentation mask, find dominant class
                    unique, counts = torch.unique(true_label, return_counts=True)
                    true_label = unique[torch.argmax(counts)]
            
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
                
                orig_img = Image.fromarray(img_np)
                
                # Build combined visualization
                width, height = orig_img.size
                heat_true = self._create_heatmap(true_cam, (width, height))
                heat_pred = self._create_heatmap(pred_cam, (width, height))
                overlay_true = Image.blend(orig_img, heat_true, alpha=0.5)
                overlay_pred = Image.blend(orig_img, heat_pred, alpha=0.5)
                
                canvas = Image.new('RGB', (width * 4, height))
                canvas.paste(orig_img, (0, 0))
                canvas.paste(heat_true, (width, 0))
                canvas.paste(overlay_true, (width * 2, 0))
                canvas.paste(heat_pred, (width * 3, 0))
                
                draw = ImageDraw.Draw(canvas)
                draw.text((10, 10), f"Original: {image_name}", fill='white')
                draw.text((width + 10, 10), f"True Class {true_label} CAM", fill='white')
                draw.text((width * 2 + 10, 10), f"True Class Overlay", fill='white')
                draw.text((width * 3 + 10, 10), f"Pred Class {pred_class} CAM", fill='white')
                
                path = self.exp_viz_dir / f'cam_{image_name}.png'
                canvas.save(path)
                logging.info(f"Saved CAM visualization for {image_name} (True: {true_label}, Pred: {pred_class})")
        
        logging.info(f"All CAM visualizations saved to {self.exp_viz_dir}")

    def _visualize_examples(self, num_samples=3):
        logging.info(f"\nSaving {num_samples} example segmentations...")
        indices = random.sample(range(len(self.dataset)), min(num_samples, len(self.dataset)))
        for idx in indices:
            sample = self.dataset[idx]
            img_t = sample['image']
            true_mask = sample['mask']
            name = sample['image_name']

            with torch.no_grad():
                out = self.model(img_t.unsqueeze(0).to(self.device))
                seg_map = out['segmentation_maps'][0].argmax(dim=0).cpu().numpy()
                
            # Unnormalize
            img_np = img_t.cpu().numpy().transpose(1, 2, 0)
            img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_np = np.clip(img_np, 0, 1)
            orig_img = Image.fromarray((img_np * 255).astype(np.uint8))
            
            # Convert masks to images
            gt_mask_img = Image.fromarray((true_mask.cpu().numpy() * 255 // true_mask.max()).astype(np.uint8)).convert('L')
            pred_mask_img = Image.fromarray((seg_map * 255 // max(1, seg_map.max())).astype(np.uint8)).convert('L')
            
            # Composite grid
            w, h = orig_img.size
            canvas = Image.new('RGB', (w * 3, h))
            canvas.paste(orig_img, (0, 0))
            canvas.paste(gt_mask_img.convert('RGB'), (w, 0))
            canvas.paste(pred_mask_img.convert('RGB'), (w * 2, 0))
            
            # Add labels
            draw = ImageDraw.Draw(canvas)
            draw.text((10, 10), "Original", fill='white')
            draw.text((w + 10, 10), "Ground Truth", fill='white')
            draw.text((w * 2 + 10, 10), "Prediction", fill='white')
            
            path = self.exp_viz_dir / f"example_{name}.png"
            canvas.save(path)
            logging.info(f"Saved example {path}")

    def _visualize_weak_supervision_comparison(self, num_samples=3):
        """Visualize the different weak supervision signals and resulting segmentation"""
        logging.info(f"\nGenerating weak supervision comparison visualizations for {num_samples} sample images...")
        
        indices = random.sample(range(len(self.dataset)), min(num_samples, len(self.dataset)))
        
        for idx in indices:
            sample = self.dataset[idx]
            image = sample['image'].unsqueeze(0).to(self.device)
            image_name = sample['image_name']
            
            # Get ground truth mask for reference
            full_mask = sample['mask']
            
            # Get weak supervision masks if available
            weak_masks = {}
            
            if 'labels' in self.weak_supervision_types:
                if 'class_label' in sample:
                    weak_masks['labels'] = f"Class Label: {sample['class_label'].item()}"
                else:
                    # If no explicit class_label, try to derive from mask
                    if full_mask.dim() <= 1:
                        weak_masks['labels'] = f"Class Label: {full_mask.item()}"
                
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
            width, height = 224, 224  # Standard size
            canvas = Image.new('RGB', (width * n_cols, height))
            
            # Add original image
            orig_img = Image.fromarray(img_np)
            canvas.paste(orig_img, (0, 0))
            
            # Add weak supervision signals
            col_idx = 1
            for sup_type, mask in weak_masks.items():
                if isinstance(mask, str):  # For class labels
                    temp_img = orig_img.copy()
                    draw = ImageDraw.Draw(temp_img)
                    draw.text((10, 10), mask, fill='white', 
                              stroke_width=1, stroke_fill='black')
                    canvas.paste(temp_img, (width * col_idx, 0))
                else:  # For visual masks
                    if mask.dim() <= 2:  # Single-channel mask
                        mask_np = mask.numpy()
                        mask_viz = np.zeros((height, width, 3), dtype=np.uint8)
                        # Use different colors for different values
                        for value in np.unique(mask_np):
                            if value == 0:  # Background or no annotation
                                color = [0, 0, 0]  # Black
                            else:
                                color = [255, 0, 0]  # Red for foreground
                            mask_viz[mask_np == value] = color
                        mask_img = Image.fromarray(mask_viz)
                    else:  # Multi-channel mask
                        # Convert to single channel by taking argmax
                        mask_np = torch.argmax(mask, dim=0).numpy()
                        # Normalize to 0-255 range
                        mask_np = mask_np * 255 // max(1, mask_np.max())
                        mask_img = Image.fromarray(mask_np.astype(np.uint8)).convert('RGB')
                    
                    canvas.paste(mask_img, (width * col_idx, 0))
                
                # Add label
                draw = ImageDraw.Draw(canvas)
                draw.text((width * col_idx + 10, 10), sup_type.capitalize(), 
                          fill='white', stroke_width=1, stroke_fill='black')
                
                col_idx += 1
            
            # Add prediction
            pred_np = predicted_mask.numpy()
            pred_viz = np.zeros((height, width, 3), dtype=np.uint8)
            # Use color coding similar to the ground truth
            for value in np.unique(pred_np):
                if value == 0:  # Background
                    color = [0, 0, 0]  # Black
                else:
                    color = [0, 255, 255]  # Cyan for predictions
                pred_viz[pred_np == value] = color
            pred_img = Image.fromarray(pred_viz)
            canvas.paste(pred_img, (width * col_idx, 0))
            draw.text((width * col_idx + 10, 10), "Prediction", 
                      fill='white', stroke_width=1, stroke_fill='black')
            col_idx += 1
            
            # Add ground truth
            gt_np = full_mask.numpy()
            gt_viz = self._colorize_mask(gt_np)
            gt_img = Image.fromarray(gt_viz)
            canvas.paste(gt_img, (width * col_idx, 0))
            draw.text((width * col_idx + 10, 10), "Ground Truth", 
                      fill='white', stroke_width=1, stroke_fill='black')
            
            # Save the visualization
            path = self.exp_viz_dir / f"weak_sup_comparison_{image_name}.png"
            canvas.save(path)
            logging.info(f"Saved weak supervision comparison for {image_name}")
        
        logging.info(f"All weak supervision comparisons saved to {self.exp_viz_dir}")
    
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
        ]
        
        # Create RGB image
        h, w = mask.shape
        colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Fill in colors for each class
        num_classes = min(len(colors), int(mask.max()) + 1)
        for i in range(num_classes):
            colored_mask[mask == i] = colors[i]
            
        return colored_mask
