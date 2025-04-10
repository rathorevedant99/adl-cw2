import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

class Evaluator:
    def __init__(self, model, dataset, config):
        self.model = model
        self.dataset = dataset
        self.config = config
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Setup data loader
        self.eval_loader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers']
        )
        
    def evaluate(self):
        self.model.eval()
        metrics = {
            'accuracy': 0,
            'mean_iou': 0,
            'pixel_accuracy': 0
        }
        
        all_preds = []
        all_labels = []
        
        print('Evaluating model...')
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.eval_loader):
                # Move data to device
                images = batch['image'].to(self.device)
                labels = batch['mask'].to(self.device)
                
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
                    labels
                )
                
                # Update metrics
                for k, v in batch_metrics.items():
                    metrics[k] += v
                
                # Store predictions and labels
                all_preds.extend(cls_preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Print progress
                if batch_idx % 10 == 0:
                    print(f'Evaluated batch {batch_idx}/{len(self.eval_loader)}')
        
        # Calculate average metrics
        num_batches = len(self.eval_loader)
        for k in metrics:
            metrics[k] /= num_batches
        
        # Calculate confusion matrix using PyTorch
        conf_matrix = self._calculate_confusion_matrix(
            torch.tensor(all_preds),
            torch.tensor(all_labels),
            num_classes=self.config['num_classes']
        )
        
        # Print metrics
        print("\nEvaluation Results:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Mean IoU: {metrics['mean_iou']:.4f}")
        print(f"Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
        print("\nConfusion Matrix:")
        print(conf_matrix)
        
        return metrics
    
    def _calculate_metrics(self, cls_preds, seg_preds, labels):
        """Calculate evaluation metrics"""
        batch_size = cls_preds.size(0)
        metrics = {
            'accuracy': 0,
            'mean_iou': 0,
            'pixel_accuracy': 0
        }
        
        for i in range(batch_size):
            # Classification accuracy
            metrics['accuracy'] += (cls_preds[i] == labels[i]).float().mean().item()
            
            # Segmentation metrics
            pred_mask = seg_preds[i]
            true_mask = labels[i]
            
            # Calculate IoU for each class
            ious = []
            for class_idx in range(self.config['num_classes']):
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