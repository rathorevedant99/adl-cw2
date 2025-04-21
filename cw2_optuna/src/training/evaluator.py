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
        self.method = config['model'].get('method','WS').upper()

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
        
        logging.info('Starting model evaluation...')
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.eval_loader):
                images = batch['image'].to(self.device)
                seg_labels = batch['mask'].to(self.device)
                if self.method == 'WS':
                    cls_labels = torch.zeros(seg_labels.size(0), dtype=torch.long, device=self.device)
                    for i in range(seg_labels.size(0)):
                        unique, counts = torch.unique(seg_labels[i], return_counts=True)
                        cls_labels[i] = unique[torch.argmax(counts)]           
            
                outputs = self.model(images)
                segmentation_maps = outputs['segmentation_maps']
                
                seg_preds = torch.argmax(segmentation_maps, dim=1)
                
                batch_metrics = self._calculate_metrics(
                    seg_preds,
                    seg_labels
                )

                metrics['mean_iou'].append(batch_metrics['mean_iou'])
                metrics['pixel_accuracy'].append(batch_metrics['pixel_accuracy'])
                if self.method == 'WS':
                    logits    = outputs['logits']
                    cls_preds = torch.argmax(logits, dim=1)
                    acc       = (cls_preds == cls_labels).float().mean().item()
                    metrics['accuracy'].append(acc)
                if self.method == 'WS':
                    all_cls_preds.extend(cls_preds.cpu().numpy())
                    all_cls_labels.extend(cls_labels.cpu().numpy())
                all_seg_preds.extend(seg_preds.cpu().numpy())
                all_seg_labels.extend(seg_labels.cpu().numpy())
                
                if batch_idx % 10 == 0:
                    logging.info(f'Evaluated batch {batch_idx}/{len(self.eval_loader)}')
        
        # final_metrics = {k: float(np.mean(v)) for k, v in metrics.items()}
        final_metrics = {}
        for k, vals in metrics.items():
            final_metrics[k] = float(np.mean(vals))
        logging.info("\nEvaluation Results:")
        if self.method == 'WS':
            logging.info(f"Classification Accuracy: {final_metrics['accuracy']:.4f}")
        logging.info(f"Mean IoU: {final_metrics['mean_iou']:.4f}")
        logging.info(f"Pixel Accuracy: {final_metrics['pixel_accuracy']:.4f}")
        
        if self.method == 'WS':
            self._visualize_cams()
        
        return final_metrics

    def _calculate_metrics(self, seg_preds, seg_labels):
        batch_size = seg_preds.size(0)
        accuracy = 0
        mean_iou = 0
        pixel_accuracy = 0
        
        for i in range(batch_size):
            pred_mask = seg_preds[i]
            true_mask = seg_labels[i]
            ious = []
            for class_idx in range(self.config['model']['num_classes']):
                pred_class = (pred_mask == class_idx) # The predicted mask for the current class
                true_class = (true_mask == class_idx) # The true mask for the current class
                intersection = (pred_class & true_class).sum().float() # The intersection of the predicted and true masks
                union = (pred_class | true_class).sum().float() # The union of the predicted and true masks
                iou = (intersection + 1e-8) / (union + 1e-8) # The IoU for the current class
                ious.append(iou.item()) # Add the IoU to the list
            mean_iou += np.mean(ious)
            pixel_accuracy += (pred_mask == true_mask).float().mean().item()
        
        return {
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
            true_label = sample['mask']
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
                
                final_img.save(self.viz_dir / f'cam_{image_name}.png')
                logging.info(f"Saved CAM visualization for {image_name} (True: {true_label}, Pred: {pred_class})")
        logging.info(f"All CAM visualizations saved to {self.viz_dir}")