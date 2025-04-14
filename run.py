#!/usr/bin/env python3
"""
Unified entry point for the weakly-supervised segmentation framework.
This script replaces the fragmented approach with a centralized CLI.
"""

import argparse
import yaml
import torch
import os
import sys
import logging
from pathlib import Path
import datetime

# Import modules from the codebase
from src.models.segmentation_model import WeaklySupervisedSegmentationModel
from src.data import PetDataset
from src.training.trainer import Trainer
from src.training.weakly_supervised_trainer import WeaklySupervisedTrainer
from src.training.evaluator import Evaluator
from src.experiments.experiment_runner import ExperimentRunner

# Setup logging
def setup_logging(log_dir, log_level=logging.INFO):
    """Setup logging configuration"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup file handler
    file_handler = logging.FileHandler(
        log_dir / f'run_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    file_handler.setFormatter(formatter)
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return root_logger

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def make_directories(config):
    """Create necessary directories for the experiment."""
    Path(config['training']['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['training']['log_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['data']['root_dir']).mkdir(parents=True, exist_ok=True)
    
    # Create directory for additional data if specified
    if 'additional_data_dir' in config.get('data', {}):
        Path(config['data']['additional_data_dir']).mkdir(parents=True, exist_ok=True)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Weakly-Supervised Semantic Segmentation Framework')
    
    # Main operation modes
    parser.add_argument('--mode', type=str, required=True, 
                      choices=['train', 'train_weak', 'eval', 'collect_data', 'experiment'],
                      help='Operation mode (train: fully supervised, train_weak: weakly supervised, ' 
                           'eval: evaluation, collect_data: data collection, experiment: run experiments)')
    
    # Configuration file
    parser.add_argument('--config', type=str, default='config.yaml',
                      help='Path to configuration file')
    
    # Common arguments
    parser.add_argument('--checkpoint', type=str, default=None,
                      help='Path to model checkpoint for evaluation or continued training')
    parser.add_argument('--download', action='store_true',
                      help='Download the Oxford-IIIT Pet dataset if not already present')
    parser.add_argument('--device', type=str, choices=['cuda', 'mps', 'cpu'], default=None,
                      help='Device to use for training/inference (overrides config)')
    parser.add_argument('--log_level', type=str, default='INFO',
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      help='Logging level')
    parser.add_argument('--output_dir', type=str, default=None,
                      help='Directory to save outputs (overrides config)')
    
    # Training specific arguments
    training_group = parser.add_argument_group('Training options')
    training_group.add_argument('--batch_size', type=int, default=None,
                              help='Batch size for training (overrides config)')
    training_group.add_argument('--num_epochs', type=int, default=None,
                              help='Number of training epochs (overrides config)')
    training_group.add_argument('--learning_rate', type=float, default=None,
                              help='Learning rate (overrides config)')
    training_group.add_argument('--weight_decay', type=float, default=None,
                              help='Weight decay for optimizer (overrides config)')
    
    # Model specific arguments
    model_group = parser.add_argument_group('Model options')
    model_group.add_argument('--backbone', type=str, default=None, 
                           choices=['resnet18', 'resnet50', 'resnet101'],
                           help='Backbone architecture (overrides config)')
    model_group.add_argument('--cam_threshold', type=float, default=None,
                           help='Threshold for class activation maps (overrides config)')
    model_group.add_argument('--region_growing_iterations', type=int, default=None,
                           help='Number of region growing iterations (overrides config)')
    
    # Weakly-supervised specific arguments
    weak_group = parser.add_argument_group('Weakly-supervised options')
    weak_group.add_argument('--use_additional_data', action='store_true',
                          help='Use additional weakly-labeled data')
    weak_group.add_argument('--additional_data_dir', type=str, default='data/additional',
                          help='Path to additional weakly-labeled data')
    weak_group.add_argument('--consistency_weight', type=float, default=None,
                          help='Weight for consistency loss (overrides config)')
    weak_group.add_argument('--curriculum_learning', action='store_true', default=None,
                          help='Use curriculum learning strategy')
    weak_group.add_argument('--no_curriculum_learning', action='store_false', dest='curriculum_learning',
                          help='Disable curriculum learning strategy')
    weak_group.add_argument('--generate_pseudo_labels', action='store_true', default=None,
                          help='Generate pseudo labels for unlabeled data')
    weak_group.add_argument('--no_pseudo_labels', action='store_false', dest='generate_pseudo_labels',
                          help='Disable pseudo label generation')
    
    # Data collection specific arguments
    data_group = parser.add_argument_group('Data collection options')
    data_group.add_argument('--flickr_key', type=str, default=None,
                          help='Flickr API key')
    data_group.add_argument('--flickr_secret', type=str, default=None,
                          help='Flickr API secret')
    data_group.add_argument('--petfinder_key', type=str, default=None,
                          help='Petfinder API key')
    data_group.add_argument('--petfinder_secret', type=str, default=None,
                          help='Petfinder API secret')
    data_group.add_argument('--max_per_source', type=int, default=500,
                          help='Maximum images to collect from each source')
    
    # Experiment specific arguments
    experiment_group = parser.add_argument_group('Experiment options')
    experiment_group.add_argument('--experiment_dir', type=str, default=None,
                                help='Directory to save experiment results')
    experiment_group.add_argument('--run_baseline', action='store_true',
                                help='Run fully-supervised baseline')
    experiment_group.add_argument('--run_ablation', action='store_true',
                                help='Run ablation study')
    
    # Evaluation specific arguments
    eval_group = parser.add_argument_group('Evaluation options')
    eval_group.add_argument('--eval_split', type=str, default='test',
                          choices=['train', 'val', 'test'],
                          help='Dataset split to evaluate on')
    eval_group.add_argument('--save_predictions', action='store_true',
                          help='Save model predictions during evaluation')
    eval_group.add_argument('--visualize', action='store_true',
                          help='Visualize model predictions during evaluation')
    
    return parser.parse_args()

def update_config_with_args(config, args):
    """Update configuration with command line arguments."""
    # Override device if specified
    if args.device:
        config['training']['device'] = args.device
    
    # Override output directories if specified
    if args.output_dir:
        output_dir = Path(args.output_dir)
        config['training']['checkpoint_dir'] = str(output_dir / 'checkpoints')
        config['training']['log_dir'] = str(output_dir / 'logs')
        config['evaluation']['checkpoint_dir'] = str(output_dir / 'checkpoints')
    
    # Override training parameters if specified
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    
    if args.num_epochs:
        config['training']['num_epochs'] = args.num_epochs
    
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    
    if args.weight_decay:
        config['training']['weight_decay'] = args.weight_decay
    
    # Override model parameters if specified
    if args.backbone:
        config['model']['backbone'] = args.backbone
    
    if args.cam_threshold is not None:
        if 'cam_threshold' not in config['model']:
            config['model']['cam_threshold'] = args.cam_threshold
        else:
            config['model']['cam_threshold'] = args.cam_threshold
    
    if args.region_growing_iterations is not None:
        if 'region_growing_iterations' not in config['model']:
            config['model']['region_growing_iterations'] = args.region_growing_iterations
        else:
            config['model']['region_growing_iterations'] = args.region_growing_iterations
    
    # Override weakly-supervised parameters if specified
    if args.consistency_weight is not None:
        if 'consistency_weight' not in config.get('training', {}):
            config['training']['consistency_weight'] = args.consistency_weight
        else:
            config['training']['consistency_weight'] = args.consistency_weight
    
    if args.curriculum_learning is not None:
        if 'curriculum_learning' not in config.get('training', {}):
            config['training']['curriculum_learning'] = args.curriculum_learning
        else:
            config['training']['curriculum_learning'] = args.curriculum_learning
    
    if args.generate_pseudo_labels is not None:
        if 'generate_pseudo_labels' not in config.get('training', {}):
            config['training']['generate_pseudo_labels'] = args.generate_pseudo_labels
        else:
            config['training']['generate_pseudo_labels'] = args.generate_pseudo_labels
    
    # Add additional data directory if using additional data
    if args.use_additional_data:
        config['data']['additional_data_dir'] = args.additional_data_dir
    
    return config

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Update configuration with command line arguments
    config = update_config_with_args(config, args)
    
    # Create required directories
    make_directories(config)
    
    # Setup logging
    log_level = getattr(logging, args.log_level)
    logger = setup_logging(config['training']['log_dir'], log_level)
    logger.info("Starting framework")
    logger.info(f"Mode: {args.mode}")
    
    # Set device
    if config['training'].get('device') == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif config['training'].get('device') == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    logger.info(f"Using device: {device}")
    
    # Handle different modes
    if args.mode == 'collect_data':
        # Import data collection module only when needed
        from src.data_collection import DataCollector
        
        logger.info("Starting data collection process")
        collector = DataCollector(
            output_dir=config['data'].get('additional_data_dir', 'data/additional'),
            flickr_api_key=args.flickr_key,
            flickr_api_secret=args.flickr_secret,
            petfinder_api_key=args.petfinder_key,
            petfinder_api_secret=args.petfinder_secret
        )
        
        collected_data = collector.collect_all_sources(max_per_source=args.max_per_source)
        
        # Output summary
        logger.info("Data collection completed")
        for source, images in collected_data.items():
            logger.info(f"Collected {len(images)} images from {source}")
    
    elif args.mode == 'train':
        # Download dataset if requested
        if args.download:
            logger.info("Downloading dataset...")
            PetDataset.download_dataset(config['data']['root_dir'])
            logger.info("Dataset download completed")
        
        # Initialize dataset
        logger.info("Initializing dataset...")
        dataset = PetDataset(
            root_dir=config['data']['root_dir'],
            split='train',
            weak_supervision=False  # Use full supervision for regular training
        )
        logger.info(f"Dataset initialized with {len(dataset)} samples")
        
        # Initialize model
        logger.info("Initializing model...")
        model = WeaklySupervisedSegmentationModel(
            num_classes=config['model']['num_classes']
        ).to(device)
        logger.info("Model initialized")
        
        # Load checkpoint if provided
        if args.checkpoint:
            logger.info(f"Loading checkpoint from {args.checkpoint}")
            checkpoint_path = Path(config['training']['checkpoint_dir']) / args.checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Checkpoint loaded")
        
        # Train model
        trainer = Trainer(
            model=model,
            dataset=dataset,
            config=config
        )
        
        logger.info("Starting training...")
        history = trainer.train()
        logger.info("Training completed")
        
        # Save final model
        final_path = Path(config['training']['checkpoint_dir']) / "final_model.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'training_history': history
        }, final_path)
        
        logger.info(f"Final model saved to {final_path}")
    
    elif args.mode == 'train_weak':
        # Download dataset if requested
        if args.download:
            logger.info("Downloading dataset...")
            PetDataset.download_dataset(config['data']['root_dir'])
            logger.info("Dataset download completed")
            
        # Initialize dataset with weak supervision
        logger.info("Initializing dataset with weak supervision...")
        original_dataset = PetDataset(
            root_dir=config['data']['root_dir'],
            split='train',
            weak_supervision=True  # Use image-level labels only
        )
        logger.info(f"Original dataset initialized with {len(original_dataset)} samples")
        
        # Initialize model with CAM-RG approach
        logger.info("Initializing model...")
        model = WeaklySupervisedSegmentationModel(
            num_classes=config['model']['num_classes'],
            cam_threshold=config['model'].get('cam_threshold', 0.2),
            region_growing_iterations=config['model'].get('region_growing_iterations', 5)
        ).to(device)
        
        # Load checkpoint if provided
        if args.checkpoint:
            logger.info(f"Loading checkpoint from {args.checkpoint}")
            checkpoint_path = Path(config['training']['checkpoint_dir']) / args.checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Checkpoint loaded")
        
        # Check for additional data
        additional_data_dir = None
        if args.use_additional_data:
            additional_data_dir = Path(args.additional_data_dir)
            if not additional_data_dir.exists() or not any(additional_data_dir.iterdir()):
                logger.warning(f"Additional data directory {additional_data_dir} does not exist or is empty.")
                logger.warning("Training will proceed with original data only.")
                additional_data_dir = None
            else:
                logger.info(f"Using additional data from {additional_data_dir}")
        
        # Setup training config
        training_config = {
            'device': str(device),
            'num_epochs': config['training']['num_epochs'],
            'batch_size': config['training']['batch_size'],
            'learning_rate': config['training']['learning_rate'],
            'weight_decay': config['training'].get('weight_decay', 1e-5),
            'checkpoint_dir': config['training']['checkpoint_dir'],
            'log_dir': config['training']['log_dir'],
            'num_classes': config['model']['num_classes'],
            'generate_pseudo_labels': config['training'].get('generate_pseudo_labels', True),
            'consistency_weight': config['training'].get('consistency_weight', 1.0),
            'curriculum_learning': config['training'].get('curriculum_learning', True),
            'strong_augment': config['training'].get('strong_augment', True),
            'save_frequency': config['training'].get('save_frequency', 5)
        }
        
        # Initialize trainer
        trainer = WeaklySupervisedTrainer(
            model=model,
            original_dataset=original_dataset,
            additional_data_dir=additional_data_dir,
            config=training_config
        )
        
        # Train model
        logger.info("Starting weakly-supervised training...")
        history = trainer.train()
        logger.info("Training completed")
        
        # Save final model
        final_path = Path(config['training']['checkpoint_dir']) / "final_model_weak.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'training_history': history
        }, final_path)
        
        logger.info(f"Final model saved to {final_path}")
    
    elif args.mode == 'eval':
        # Ensure checkpoint is provided
        if not args.checkpoint:
            logger.error("Checkpoint path must be provided for evaluation mode")
            return
        
        # Download dataset if requested
        if args.download:
            logger.info("Downloading dataset...")
            PetDataset.download_dataset(config['data']['root_dir'])
            logger.info("Dataset download completed")
            
        # Initialize dataset
        logger.info(f"Initializing evaluation dataset (split: {args.eval_split})...")
        dataset = PetDataset(
            root_dir=config['data']['root_dir'],
            split=args.eval_split,
            weak_supervision=False  # Always use full supervision for evaluation
        )
        logger.info(f"Dataset initialized with {len(dataset)} samples")
        
        # Initialize model
        logger.info("Initializing model...")
        model = WeaklySupervisedSegmentationModel(
            num_classes=config['model']['num_classes'],
            cam_threshold=config['model'].get('cam_threshold', 0.2),
            region_growing_iterations=config['model'].get('region_growing_iterations', 5)
        ).to(device)
        
        # Load checkpoint
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        checkpoint_path = Path(config['evaluation']['checkpoint_dir']) / args.checkpoint
        if not checkpoint_path.exists() and not os.path.isabs(args.checkpoint):
            # Try as absolute path
            checkpoint_path = Path(args.checkpoint)
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Checkpoint loaded")
        
        # Initialize evaluator
        evaluator = Evaluator(
            model=model,
            dataset=dataset,
            config=config
        )
        
        # Evaluate model
        logger.info("Evaluating model...")
        metrics = evaluator.evaluate(
            save_predictions=args.save_predictions,
            visualize=args.visualize
        )
        
        # Report metrics
        logger.info("Evaluation completed")
        logger.info("Metrics:")
        for metric_name, metric_value in metrics.items():
            logger.info(f"  {metric_name}: {metric_value}")
    
    elif args.mode == 'experiment':
        # Run experiments
        logger.info("Running experiments")
        
        # Initialize experiment runner
        runner = ExperimentRunner(args.config, args.experiment_dir)
        
        # Run fully-supervised baseline if requested
        if args.run_baseline:
            runner.run_fully_supervised_baseline()
        
        # Get experiment configurations
        from src.experiments.experiment_runner import setup_experiment_configs
        experiment_configs = setup_experiment_configs()
        
        # Run each experiment
        for name, config_overrides in experiment_configs.items():
            runner.run_weakly_supervised_experiment(
                name,
                config_overrides=config_overrides,
                use_additional_data=args.use_additional_data,
                additional_data_dir=args.additional_data_dir
            )
        
        # Run ablation study if requested
        if args.run_ablation:
            runner.run_ablation_study()
        
        # Compare all experiment results
        runner.compare_all_experiments()
        
        logger.info("Experiments completed")
    
    else:
        logger.error(f"Unknown mode: {args.mode}")
        return

if __name__ == '__main__':
    main()
