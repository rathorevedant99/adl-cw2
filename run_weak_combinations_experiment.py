"""
GenAI use declaration: Scaffold for this script was generated using GenAI tools (Anthropic's Claude)
"""

import argparse
import yaml
import torch
from pathlib import Path
import logging
import sys
import datetime
import itertools
import csv
import numpy as np
import matplotlib.pyplot as plt
from src.data import PetDataset
from src.models.segmentation_model_resnet50 import WeaklySupervisedSegmentationModelResNet50
from src.models.segmentation_model_unet import WeaklySupervisedSegmentationModelUNet
from src.training.trainer import Trainer
from src.training.evaluator import Evaluator
import torchvision.transforms as T


def setup_logging(config, experiment_name="weak_combination_experiment"):
    """Setup logging configuration with experiment-specific name"""
    log_dir = Path(config['training']['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup file handler
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(log_dir / f'{experiment_name}_{timestamp}.log')
    file_handler.setFormatter(formatter)
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return root_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Weak Supervision Combination Experiment')
    parser.add_argument('--config', type=str, default='config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--download', action='store_true',
                      help='Download the dataset if not already present')
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet50', 'unet'],
                      help='Backbone model to use for experiments')
    parser.add_argument('--iterations', type=int, default=1,
                      help='Number of iterations for each combination')
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
    
def make_directories(config, experiment_root="experiments/weak_combinations"):
    """Create experiment-specific directories"""
    Path(config['training']['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['training']['log_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['data']['root_dir']).mkdir(parents=True, exist_ok=True)
    
    # Create experiment-specific directories
    experiment_dir = Path(experiment_root)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    for subdir in ['checkpoints', 'logs', 'results', 'visualizations']:
        (experiment_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    return experiment_dir


def run_experiment(config, weak_supervision_types, backbone, experiment_dir, iteration):
    """Run a single experiment with the specified weak supervision types"""
    experiment_name = f"{'_'.join(weak_supervision_types)}_{iteration}"
    logger = logging.getLogger()
    logger.info(f"Starting experiment with weak supervision types: {weak_supervision_types}, iteration: {iteration}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create transform pipeline
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225])
    ])
    
    # Add this before creating the datasets in run_experiment function
    from src.data import PetDataset
    PetDataset._create_split_files_static(config['data']['root_dir'], test_split=0.2)

    # Initialize datasets with specific weak supervision types
    train_dataset = PetDataset(
        root_dir=config['data']['root_dir'],
        split='train',
        weak_supervision=True,
        weak_supervision_types=weak_supervision_types,
        transform=transform,
        subset_fraction=0.2
    )
    
    val_dataset = PetDataset(
        root_dir=config['data']['root_dir'],
        split='val',
        weak_supervision=True,
        weak_supervision_types=weak_supervision_types,
        transform=transform,
        subset_fraction=0.2
    )
    
    test_dataset = PetDataset(
        root_dir=config['data']['root_dir'],
        split='test',
        weak_supervision=True,
        weak_supervision_types=weak_supervision_types,
        transform=transform,
        subset_fraction=0.2
    )
    
    # Initialize model
    logger.info(f"Initializing {backbone} model...")
    if backbone == 'resnet50':
        model = WeaklySupervisedSegmentationModelResNet50(
            num_classes=config['model']['num_classes'],
        ).to(device)
    elif backbone == 'unet':
        model = WeaklySupervisedSegmentationModelUNet(
            num_classes=config['model']['num_classes'],
        ).to(device)
    
    # Setup experiment-specific config
    experiment_config = config.copy()
    experiment_config['training']['checkpoint_dir'] = str(experiment_dir / 'checkpoints' / experiment_name)
    experiment_config['training']['log_dir'] = str(experiment_dir / 'logs' / experiment_name)
    experiment_config['training']['num_epochs'] = min(config['training']['num_epochs'], 25)  # Limit epochs for experimentation
    
    # Create directories
    Path(experiment_config['training']['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    Path(experiment_config['training']['log_dir']).mkdir(parents=True, exist_ok=True)
    
    # Train model
    logger.info(f"Training model with weak supervision types: {weak_supervision_types}")
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=experiment_config,
        weak_supervision_types=weak_supervision_types
    )
    trainer.train()
    
    # Evaluate model
    logger.info(f"Evaluating model with weak supervision types: {weak_supervision_types}")
    evaluator = Evaluator(
        model=model,
        dataset=test_dataset,
        config=experiment_config,
        weak_supervision_types=weak_supervision_types
    )
    metrics = evaluator.evaluate()
    
    return {
        'weak_supervision_types': '_'.join(weak_supervision_types),
        'iteration': iteration,
        'accuracy': metrics['accuracy'],
        'mean_iou': metrics['mean_iou'],
        'pixel_accuracy': metrics['pixel_accuracy']
    }


def visualize_results(results, experiment_dir):
    """Create visualizations of the experiment results without using pandas"""
    # Group results by weak_supervision_types
    grouped_results = {}
    for result in results:
        key = result['weak_supervision_types']
        if key not in grouped_results:
            grouped_results[key] = []
        grouped_results[key].append(result)
    
    # Calculate summary statistics
    summary = []
    for key, group in grouped_results.items():
        mean_iou = sum(item['mean_iou'] for item in group) / len(group)
        mean_pixel_accuracy = sum(item['pixel_accuracy'] for item in group) / len(group)
        mean_accuracy = sum(item['accuracy'] for item in group) / len(group)
        summary.append({
            'weak_supervision_types': key,
            'mean_iou': mean_iou,
            'mean_pixel_accuracy': mean_pixel_accuracy,
            'mean_accuracy': mean_accuracy
        })
    
    # Sort summary by mean_iou in descending order
    summary.sort(key=lambda x: x['mean_iou'], reverse=True)
    
    # Create a bar plot for mean IoU
    plt.figure(figsize=(12, 6))
    keys = [item['weak_supervision_types'] for item in summary]
    values = [item['mean_iou'] for item in summary]
    plt.bar(keys, values)
    plt.title('Mean IoU by Weak Supervision Combination')
    plt.xlabel('Weak Supervision Types')
    plt.ylabel('Mean IoU')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(experiment_dir / 'results' / 'mean_iou_comparison.png')
    
    # Create a bar plot for pixel accuracy
    plt.figure(figsize=(12, 6))
    values = [item['mean_pixel_accuracy'] for item in summary]
    plt.bar(keys, values)
    plt.title('Pixel Accuracy by Weak Supervision Combination')
    plt.xlabel('Weak Supervision Types')
    plt.ylabel('Pixel Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(experiment_dir / 'results' / 'pixel_accuracy_comparison.png')
    
    # Create a box plot to show variation across iterations
    plt.figure(figsize=(14, 8))
    data = [
        [result['mean_iou'] for result in grouped_results[key]]
        for key in keys
    ]
    plt.boxplot(data, labels=keys)
    plt.title('Mean IoU Distribution by Weak Supervision Combination')
    plt.xlabel('Weak Supervision Types')
    plt.ylabel('Mean IoU')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(experiment_dir / 'results' / 'mean_iou_boxplot.png')
    
    return summary


def main():
    args = parse_args()
    config = load_config(args.config)
    experiment_dir = make_directories(config)
    
    # Setup logging
    logger = setup_logging(config)
    logger.info("Starting weak supervision combination experiments")
    logger.info(f"Backbone: {args.backbone}, Iterations: {args.iterations}")
    
    # Define weak supervision types to experiment with
    weak_types = ['labels', 'bboxes', 'scribbles']
    combinations = []
    
    # Generate all possible combinations of weak supervision types
    for r in range(1, len(weak_types) + 1):
        combinations.extend(list(itertools.combinations(weak_types, r)))
    
    # Convert combinations to lists for easier handling
    combinations = [list(combo) for combo in combinations]
    
    # Download dataset if requested
    if args.download:
        logger.info("Downloading dataset...")
        PetDataset.download_dataset(config['data']['root_dir'])
        logger.info("Dataset download completed")
    
    # Run experiments for each combination
    results = []
    for combo in combinations:
        for i in range(args.iterations):
            result = run_experiment(config, combo, args.backbone, experiment_dir, i+1)
            results.append(result)
            
            # Save intermediate results
            with open(experiment_dir / 'results' / 'experiment_results.csv', 'w', newline='') as f:
                fieldnames = ['weak_supervision_types', 'iteration', 'accuracy', 'mean_iou', 'pixel_accuracy']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
    
    # Analyze and visualize results
    summary = visualize_results(results, experiment_dir)
    
    # Print best combination
    best_combo = summary[0]['weak_supervision_types']
    best_iou = summary[0]['mean_iou']
    logger.info(f"\nExperiment completed. Best weak supervision combination: {best_combo} with mean IoU: {best_iou:.4f}")
    
    # Save final results
    with open(experiment_dir / 'results' / 'experiment_results.csv', 'w', newline='') as f:
        fieldnames = ['weak_supervision_types', 'iteration', 'accuracy', 'mean_iou', 'pixel_accuracy']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
        
    with open(experiment_dir / 'results' / 'experiment_summary.csv', 'w', newline='') as f:
        fieldnames = ['weak_supervision_types', 'mean_iou', 'mean_pixel_accuracy', 'mean_accuracy']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)
    
    logger.info(f"Results saved to {experiment_dir / 'results'}")


if __name__ == '__main__':
    main()
