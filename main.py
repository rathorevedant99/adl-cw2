import argparse
import yaml
import torch
from pathlib import Path
import logging
import sys
import datetime

from src.models.segmentation_model import WeaklySupervisedSegmentationModel
from src.data import PetDataset
from src.training.trainer import Trainer
from src.training.evaluator import Evaluator


def setup_logging(config):
    """Setup logging configuration"""
    log_dir = Path(config['training']['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup file handler
    file_handler = logging.FileHandler(log_dir / f'run_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
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
    parser = argparse.ArgumentParser(description='Weakly-Supervised Semantic Segmentation')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'eval'],
                      help='Mode to run the model in')
    parser.add_argument('--config', type=str, default='config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str,
                      help='Path to model checkpoint for evaluation')
    parser.add_argument('--download', action='store_true',
                      help='Download the dataset if not already present')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
def make_directories(config):
    """Create directories for the experiment"""
    Path(config['training']['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['training']['log_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['data']['root_dir']).mkdir(parents=True, exist_ok=True)


def main():
    args = parse_args()
    config = load_config(args.config)
    make_directories(config)
    
    # Setup logging
    logger = setup_logging(config)
    logger.info("Starting pipeline")
    logger.info(f"Mode: {args.mode}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if args.download:
        logger.info("Downloading dataset...")
        PetDataset.download_dataset(config['data']['root_dir'])
        logger.info("Dataset download completed")
    
    # Initialize dataset
    logger.info("Initializing dataset...")
    dataset = PetDataset(
        root_dir=config['data']['root_dir'],
        split='train' if args.mode == 'train' else 'val',
        weak_supervision=True
    )
    logger.info(f"Dataset initialized with {len(dataset)} samples")
    
    # Initialize model
    logger.info("Initializing model...")
    model = WeaklySupervisedSegmentationModel(
        num_classes=config['model']['num_classes'],
        backbone=config['model']['backbone']
    ).to(device)
    logger.info("Model initialized")
    
    if args.mode == 'train':
        logger.info("Starting training...")
        trainer = Trainer(
            model=model,
            dataset=dataset,
            config=config
        )
        trainer.train()
        logger.info("Training completed")
    else:
        if not args.checkpoint:
            raise ValueError("Checkpoint path must be provided for evaluation mode")
        
        # Load checkpoint
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        checkpoint_path = Path(config['evaluation']['checkpoint_dir']) / args.checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Checkpoint loaded")
        
        logger.info("Starting evaluation...")
        evaluator = Evaluator(
            model=model,
            dataset=dataset,
            config=config
        )
        metrics = evaluator.evaluate()
        logger.info("Evaluation completed")
        logger.info("Evaluation metrics:")
        for metric_name, value in metrics.items():
            logger.info(f"{metric_name}: {value:.4f}")

if __name__ == '__main__':
    main()
