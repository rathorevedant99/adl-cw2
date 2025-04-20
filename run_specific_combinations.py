import argparse
import yaml
import torch
from pathlib import Path
import logging
import sys
import datetime
import csv
from src.data import PetDataset
from src.models.segmentation_model_resnet50 import WeaklySupervisedSegmentationModelResNet50
from src.models.segmentation_model_unet import WeaklySupervisedSegmentationModelUNet
from src.training.trainer import Trainer
from src.training.evaluator import Evaluator
from src.experiment_utils import compare_weak_supervision_results, visualize_samples_with_weak_supervision
import torchvision.transforms as T

def setup_logging(config, experiment_name="specific_combinations"):
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
    parser = argparse.ArgumentParser(description='Specific Weak Supervision Combinations Experiment')
    parser.add_argument('--config', type=str, default='config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--download', action='store_true',
                      help='Download the dataset if not already present')
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet50', 'unet'],
                      help='Backbone model to use for experiments')
    parser.add_argument('--visualize_samples', action='store_true',
                      help='Visualize samples with different weak supervision signals')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
def make_directories(experiment_dir="experiments/specific_combinations"):
    """Create experiment-specific directories"""
    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    for subdir in ['checkpoints', 'logs', 'results', 'visualizations', 'samples']:
        (experiment_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    return experiment_dir

def run_experiment(config, weak_supervision_types, backbone, experiment_dir):
    """Run a single experiment with the specified weak supervision types"""
    experiment_name = f"{'_'.join(weak_supervision_types)}"
    logger = logging.getLogger()
    logger.info(f"Starting experiment with weak supervision types: {weak_supervision_types}")
    
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
    
    # Initialize datasets with specific weak supervision types
    train_dataset = PetDataset(
        root_dir=config['data']['root_dir'],
        split='train',
        weak_supervision=True,
        weak_supervision_types=weak_supervision_types,
        transform=transform
    )
    
    val_dataset = PetDataset(
        root_dir=config['data']['root_dir'],
        split='val',
        weak_supervision=True,
        weak_supervision_types=weak_supervision_types,
        transform=transform
    )
    
    test_dataset = PetDataset(
        root_dir=config['data']['root_dir'],
        split='test',
        weak_supervision=True,
        weak_supervision_types=weak_supervision_types,
        transform=transform
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
    experiment_config['training']['num_epochs'] = 10  # Shorter to demonstrate concept
    
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
        'accuracy': metrics['accuracy'],
        'mean_iou': metrics['mean_iou'],
        'pixel_accuracy': metrics['pixel_accuracy']
    }

def main():
    args = parse_args()
    config = load_config(args.config)
    experiment_dir = make_directories()
    
    # Setup logging
    logger = setup_logging(config)
    logger.info("Starting specific weak supervision combinations experiment")
    logger.info(f"Backbone: {args.backbone}")
    
    # Define the specific combinations to test
    combinations = [
        ['labels'],
        ['bboxes'],
        ['scribbles'],
        ['labels', 'bboxes'],
        ['labels', 'scribbles'],
        ['bboxes', 'scribbles'],
        ['labels', 'bboxes', 'scribbles']
    ]
    
    # Download dataset if requested
    if args.download:
        logger.info("Downloading dataset...")
        PetDataset.download_dataset(config['data']['root_dir'])
        logger.info("Dataset download completed")
    
    # Visualize samples with different weak supervision if requested
    if args.visualize_samples:
        logger.info("Visualizing samples with different weak supervision signals...")
        # Need to create a dataset with all weak supervision types
        all_weak_types = ['labels', 'bboxes', 'scribbles']
        sample_dataset = PetDataset(
            root_dir=config['data']['root_dir'],
            split='train',
            weak_supervision=True,
            weak_supervision_types=all_weak_types,
            transform=None  # Use default transform
        )
        visualize_samples_with_weak_supervision(
            sample_dataset, 
            num_samples=5, 
            output_dir=experiment_dir / 'samples'
        )
    
    # Run experiments for each combination
    results = []
    for combo in combinations:
        result = run_experiment(config, combo, args.backbone, experiment_dir)
        results.append(result)
            
        # Save intermediate results
        with open(experiment_dir / 'results' / 'experiment_results.csv', 'w', newline='') as f:
            fieldnames = ['weak_supervision_types', 'accuracy', 'mean_iou', 'pixel_accuracy']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
    
    # Analyze and visualize results
    summary = compare_weak_supervision_results(results, experiment_dir / 'results')
    
    # Print best combination
    best_combo = summary[0]['weak_supervision_types']
    best_iou = summary[0]['mean_iou']
    logger.info(f"\nExperiment completed. Best weak supervision combination: {best_combo} with mean IoU: {best_iou:.4f}")
    
    # Save final results
    with open(experiment_dir / 'results' / 'experiment_results.csv', 'w', newline='') as f:
        fieldnames = ['weak_supervision_types', 'accuracy', 'mean_iou', 'pixel_accuracy']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    logger.info(f"Results saved to {experiment_dir / 'results'}")
    logger.info(f"Experiment completed successfully!")

if __name__ == '__main__':
    main()