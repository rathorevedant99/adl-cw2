import torch
import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path
import logging
from PIL import Image, ImageDraw, ImageFont
import json

def compare_weak_supervision_results(results, output_dir):
    """
    Compare and visualize results from different weak supervision experiments without using pandas
    
    Args:
        results: List of dictionaries with experiment results
        output_dir: Directory to save visualizations
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Group results by weak_supervision_types
    grouped_results = {}
    for result in results:
        key = result['weak_supervision_types']
        if key not in grouped_results:
            grouped_results[key] = []
        grouped_results[key].append(result)
    
    # Compute summary statistics by supervision type
    summary = []
    for key, group in grouped_results.items():
        # Calculate stats for mean_iou
        iou_values = [item['mean_iou'] for item in group]
        mean_iou = sum(iou_values) / len(iou_values)
        std_iou = np.std(iou_values) if len(iou_values) > 1 else 0
        min_iou = min(iou_values)
        max_iou = max(iou_values)
        
        # Calculate stats for pixel_accuracy
        pixel_acc_values = [item['pixel_accuracy'] for item in group]
        mean_pixel_acc = sum(pixel_acc_values) / len(pixel_acc_values)
        std_pixel_acc = np.std(pixel_acc_values) if len(pixel_acc_values) > 1 else 0
        min_pixel_acc = min(pixel_acc_values)
        max_pixel_acc = max(pixel_acc_values)
        
        # Calculate stats for accuracy
        acc_values = [item['accuracy'] for item in group]
        mean_acc = sum(acc_values) / len(acc_values)
        std_acc = np.std(acc_values) if len(acc_values) > 1 else 0
        min_acc = min(acc_values)
        max_acc = max(acc_values)
        
        summary.append({
            'weak_supervision_types': key,
            'mean_iou': mean_iou,
            'std_iou': std_iou,
            'min_iou': min_iou,
            'max_iou': max_iou,
            'mean_pixel_accuracy': mean_pixel_acc,
            'std_pixel_accuracy': std_pixel_acc,
            'min_pixel_accuracy': min_pixel_acc,
            'max_pixel_accuracy': max_pixel_acc,
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'min_accuracy': min_acc,
            'max_accuracy': max_acc
        })
    
    # Sort by mean IoU for better visualization
    summary.sort(key=lambda x: x['mean_iou'], reverse=True)
    
    # Extract data for plotting
    keys = [item['weak_supervision_types'] for item in summary]
    iou_means = [item['mean_iou'] for item in summary]
    iou_stds = [item['std_iou'] for item in summary]
    pixel_acc_means = [item['mean_pixel_accuracy'] for item in summary]
    pixel_acc_stds = [item['std_pixel_accuracy'] for item in summary]
    
    # Create barplot for Mean IoU
    plt.figure(figsize=(10, 6))
    x = np.arange(len(keys))
    plt.bar(x, iou_means)
    plt.errorbar(
        x=x, 
        y=iou_means,
        yerr=iou_stds,
        fmt='none', 
        capsize=5, 
        color='black'
    )
    plt.title('Mean IoU by Weak Supervision Type')
    plt.xlabel('Weak Supervision Types')
    plt.ylabel('Mean IoU')
    plt.xticks(x, keys, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'mean_iou_comparison.png')
    plt.close()
    
    # Create barplot for Pixel Accuracy
    plt.figure(figsize=(10, 6))
    plt.bar(x, pixel_acc_means)
    plt.errorbar(
        x=x, 
        y=pixel_acc_means,
        yerr=pixel_acc_stds,
        fmt='none', 
        capsize=5, 
        color='black'
    )
    plt.title('Pixel Accuracy by Weak Supervision Type')
    plt.xlabel('Weak Supervision Types')
    plt.ylabel('Pixel Accuracy')
    plt.xticks(x, keys, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'pixel_accuracy_comparison.png')
    plt.close()
    
    # Create boxplots to show distribution
    plt.figure(figsize=(12, 6))
    data = [
        [result['mean_iou'] for result in grouped_results[key]]
        for key in keys
    ]
    plt.boxplot(data, labels=keys)
    plt.title('Mean IoU Distribution by Weak Supervision Type')
    plt.xlabel('Weak Supervision Types')
    plt.ylabel('Mean IoU')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'mean_iou_boxplot.png')
    plt.close()
    
    # Save summary as CSV
    with open(output_dir / 'weak_supervision_summary.csv', 'w', newline='') as f:
        fieldnames = [
            'weak_supervision_types', 
            'mean_iou', 'std_iou', 'min_iou', 'max_iou',
            'mean_pixel_accuracy', 'std_pixel_accuracy', 'min_pixel_accuracy', 'max_pixel_accuracy',
            'mean_accuracy', 'std_accuracy', 'min_accuracy', 'max_accuracy'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)
    
    # Create a markdown report
    with open(output_dir / 'experiment_report.md', 'w') as f:
        f.write("# Weak Supervision Experiment Results\n\n")
        
        f.write("## Summary\n\n")
        f.write("This report compares different combinations of weak supervision types for semantic segmentation.\n\n")
        
        f.write("## Mean IoU Comparison\n\n")
        f.write("| Weak Supervision Types | Mean IoU | Std Dev | Min | Max |\n")
        f.write("|------------------------|----------|---------|-----|-----|\n")
        for item in summary:
            f.write(f"| {item['weak_supervision_types']} | {item['mean_iou']:.4f} | {item['std_iou']:.4f} | {item['min_iou']:.4f} | {item['max_iou']:.4f} |\n")
        
        f.write("\n## Pixel Accuracy Comparison\n\n")
        f.write("| Weak Supervision Types | Pixel Accuracy | Std Dev | Min | Max |\n")
        f.write("|------------------------|----------------|---------|-----|-----|\n")
        for item in summary:
            f.write(f"| {item['weak_supervision_types']} | {item['mean_pixel_accuracy']:.4f} | {item['std_pixel_accuracy']:.4f} | {item['min_pixel_accuracy']:.4f} | {item['max_pixel_accuracy']:.4f} |\n")
        
        f.write("\n## Classification Accuracy Comparison\n\n")
        f.write("| Weak Supervision Types | Classification Accuracy | Std Dev | Min | Max |\n")
        f.write("|------------------------|-----------------------|---------|-----|-----|\n")
        for item in summary:
            f.write(f"| {item['weak_supervision_types']} | {item['mean_accuracy']:.4f} | {item['std_accuracy']:.4f} | {item['min_accuracy']:.4f} | {item['max_accuracy']:.4f} |\n")
        
        f.write("\n## Conclusion\n\n")
        best_item = summary[0]
        f.write(f"The best performing weak supervision combination was **{best_item['weak_supervision_types']}** with a mean IoU of {best_item['mean_iou']:.4f} and pixel accuracy of {best_item['mean_pixel_accuracy']:.4f}.\n")
    
    logging.info(f"Experiment report saved to {output_dir / 'experiment_report.md'}")
    return summary

def visualize_samples_with_weak_supervision(dataset, num_samples=5, output_dir="weak_supervision_samples"):
    """
    Visualize samples from the dataset with different weak supervision signals
    
    Args:
        dataset: PetDataset instance
        num_samples: Number of samples to visualize
        output_dir: Directory to save visualizations
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    for idx in indices:
        sample = dataset[idx]
        img_name = sample['image_name']
        
        # Convert image tensor to numpy array
        img = sample['image'].numpy().transpose(1, 2, 0)
        img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
        img = img.astype(np.uint8)
        
        # Get ground truth mask
        gt_mask = sample['full_mask'].numpy()
        
        # Create figure with subplots
        n_cols = 2  # Original image and GT mask
        
        # Check for available weak supervision signals
        if 'bbox_mask' in sample:
            n_cols += 1
        if 'scribble_mask' in sample:
            n_cols += 1
        
        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
        
        # Plot original image
        axes[0].imshow(img)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # Plot ground truth mask
        gt_vis = colorize_mask(gt_mask)
        axes[1].imshow(gt_vis)
        axes[1].set_title("Ground Truth Mask")
        axes[1].axis('off')
        
        col_idx = 2
        
        # Plot bounding box if available
        if 'bbox_mask' in sample:
            bbox_mask = sample['bbox_mask'].numpy()
            # Create RGB image
            bbox_vis = np.zeros((224, 224, 3), dtype=np.uint8)
            # Make it mostly transparent
            bbox_vis[bbox_mask > 0] = [255, 0, 0]  # Red for bbox
            
            # Overlay bbox on image
            bbox_img = img.copy()
            alpha = 0.3
            bbox_img = np.uint8(bbox_img * (1 - alpha) + bbox_vis * alpha)
            
            axes[col_idx].imshow(bbox_img)
            axes[col_idx].set_title("Bounding Box")
            axes[col_idx].axis('off')
            col_idx += 1
        
        # Plot scribbles if available
        if 'scribble_mask' in sample:
            scribble_mask = sample['scribble_mask'].numpy()
            scribble_vis = colorize_mask(scribble_mask)
            
            # Overlay scribbles on image
            scribble_img = img.copy()
            mask = (scribble_mask > 0)
            alpha = 0.6
            for c in range(3):
                scribble_img[:, :, c] = np.where(
                    mask, 
                    np.uint8(img[:, :, c] * (1 - alpha) + scribble_vis[:, :, c] * alpha),
                    img[:, :, c]
                )
            
            axes[col_idx].imshow(scribble_img)
            axes[col_idx].set_title("Scribbles")
            axes[col_idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / f"{img_name}_weak_supervisions.png")
        plt.close()
    
    logging.info(f"Saved {num_samples} sample visualizations to {output_dir}")

def colorize_mask(mask):
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

def create_experiment_config(base_config, experiment_name, weak_supervision_types=None, 
                            backbone='resnet50', num_epochs=25):
    """
    Create experiment-specific configuration
    
    Args:
        base_config: Base configuration dictionary
        experiment_name: Name of the experiment
        weak_supervision_types: List of weak supervision types to use
        backbone: Backbone model to use
        num_epochs: Number of epochs to train for
        
    Returns:
        experiment_config: Experiment-specific configuration
    """
    experiment_config = base_config.copy()
    
    # Set experiment-specific directories
    experiment_dir = Path(f"experiments/weak_combinations/{experiment_name}")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Update directories
    experiment_config['training']['checkpoint_dir'] = str(experiment_dir / 'checkpoints')
    experiment_config['training']['log_dir'] = str(experiment_dir / 'logs')
    experiment_config['evaluation']['checkpoint_dir'] = str(experiment_dir / 'checkpoints')
    experiment_config['viz_dir'] = str(experiment_dir / 'visualizations')
    
    # Create directories
    Path(experiment_config['training']['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    Path(experiment_config['training']['log_dir']).mkdir(parents=True, exist_ok=True)
    Path(experiment_config['viz_dir']).mkdir(parents=True, exist_ok=True)
    
    # Update model configuration
    experiment_config['model']['backbone'] = backbone
    
    # Update training configuration
    experiment_config['training']['num_epochs'] = num_epochs
    
    # Add weak supervision types
    experiment_config['weak_supervision_types'] = weak_supervision_types
    
    # Save experiment configuration
    with open(experiment_dir / 'config.json', 'w') as f:
        json.dump(experiment_config, f, indent=2)
    
    return experiment_config