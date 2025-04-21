import argparse
import yaml
import torch
from pathlib import Path
import logging
import sys
import datetime
import torchvision.transforms as T
from torch.utils.data import ConcatDataset
import shutil

from src.models.segmentation_model_resnet50 import WeaklySupervisedSegmentationModelResNet50, FullySupervisedSegmentationModelResNet50
from src.models.segmentation_model_unet import WeaklySupervisedSegmentationModelUNet, FullySupervisedSegmentationModelUNet
from src.data import PetDataset
from src.training.trainer import Trainer
from src.training.evaluator import Evaluator
from src.augment import AugmentedDataset


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
    
    method = config['model'].get('method', 'WS').upper()
    if method not in ('FS', 'WS'):
        raise ValueError(f"Unsupported method: {method}. Choose FS or WS.")
    if method == 'FS':
        config['model']['num_classes'] = 2
    
    is_weak = (method == 'WS')
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Check if dataset exists and download if needed
    data_path = Path(config['data']['root_dir'])
    images_dir = data_path / 'images'
    annotations_dir = data_path / 'annotations'
    
    # Check if dataset exists
    if args.download or not images_dir.exists() or not annotations_dir.exists() or len(list(images_dir.glob('*.jpg'))) == 0:
        logger.info("Dataset not found or download flag set. Downloading...")
        PetDataset.download_dataset(config['data']['root_dir'])
        logger.info("Dataset download and split completed")
        
        # After downloading, check if masks are in the right location
        trimaps_dir = annotations_dir / 'trimaps'
        if trimaps_dir.exists() and not any(annotations_dir.glob('*.png')):
            logger.info("Moving mask files from trimaps to annotations directory...")
            for png_file in trimaps_dir.glob('*.png'):
                shutil.copy(png_file, annotations_dir / png_file.name)
            logger.info("Mask files moved successfully.")
    else:
        logger.info("Dataset found. Skipping download.")
        
        # Still check for masks in the right location
        trimaps_dir = annotations_dir / 'trimaps'
        if trimaps_dir.exists() and not any(annotations_dir.glob('*.png')):
            logger.info("Moving mask files from trimaps to annotations directory...")
            for png_file in trimaps_dir.glob('*.png'):
                shutil.copy(png_file, annotations_dir / png_file.name)
            logger.info("Mask files moved successfully.")
    
    # Create basic transform pipeline
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225])
    ])
    
    if args.mode == 'train':
        logger.info("Initializing training and validation datasets...")
        train_dataset = PetDataset(
            root_dir=config['data']['root_dir'],
            split='train',
            weak_supervision=is_weak,
            transform=transform
        )
        val_dataset = PetDataset(
            root_dir=config['data']['root_dir'],
            split='val',
            weak_supervision=is_weak,
            transform=transform
        )

        if method == 'WS':
            if config['training']['use_augmentation']:
                logger.info("Creating augmented dataset...")
                augmented = AugmentedDataset(train_dataset)
                augmented._build_augmented_indices()
                augmented.save_sample_pairs(
                num_samples=5,
                save_dir=Path(config['training']['log_dir']) / 'augmentation_samples'
            )
                full_train_dataset = ConcatDataset([train_dataset, augmented])
                logger.info(f"Combined dataset size: {len(full_train_dataset)} (original: {len(train_dataset)}, augmented: {len(augmented)})")
                logger.info("Saving sample pairs of original and augmented images...")
        else:
            full_train_dataset = train_dataset
            logger.info(f"Training dataset size: {len(train_dataset)} samples")

    else:
        logger.info("Initializing test dataset for evaluation...")
        if is_weak:
            # 1) Dataset that returns the image‐level class label
            weak_ds = PetDataset(
                root_dir         = config['data']['root_dir'],
                split            = 'test',
                weak_supervision = True,     # gives you batch['mask'] = class_idx
                transform        = transform
            )
            # 2) Dataset that returns the full pixel mask
            fs_ds = PetDataset(
                root_dir         = config['data']['root_dir'],
                split            = 'test',
                weak_supervision = False,    # gives you batch['mask'] = H×W mask
                transform        = transform
            )
            # 3) Zip them together
            eval_ds = list(zip(weak_ds, fs_ds))
        else:
            # Fully‐supervised: you only need the one dataset
            eval_ds = PetDataset(
                root_dir         = config['data']['root_dir'],
                split            = 'test',
                weak_supervision = False,
                transform        = transform
            )
        logger.info(f"Test dataset size: {len(eval_ds)} samples")                
    # Initialize model
    logger.info("Initializing model...")
    # choose FS vs WS
    if method =='WS':
        if config['model']['backbone'] == 'resnet50':
            model = WeaklySupervisedSegmentationModelResNet50(
                num_classes=config['model']['num_classes'],
            ).to(device)
        elif config['model']['backbone'] == 'unet':
            model = WeaklySupervisedSegmentationModelUNet(
                num_classes=config['model']['num_classes'],
            ).to(device)
        else:
            raise ValueError(f"Unsupported backbone: {config['model']['backbone']}")
    elif method == 'FS':
        if config['model']['backbone'] == 'resnet50':
            model = FullySupervisedSegmentationModelResNet50(
                num_classes=config['model']['num_classes'],
            ).to(device)
        elif config['model']['backbone'] == 'unet':
            model = FullySupervisedSegmentationModelUNet(
                num_classes=config['model']['num_classes'],
            ).to(device)
        else:
            raise ValueError(f"Unsupported backbone: {config['model']['backbone']}")
    else:
        raise ValueError(f"Unsupported supervised method: {config['model']['method']}")
    logger.info("Model initialized")

    
    if args.mode == 'train':
        logger.info("Starting training...")
        trainer = Trainer(
            model=model,
            train_dataset=full_train_dataset,
            val_dataset=val_dataset,
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
            dataset=eval_ds,
            config=config
        )
        metrics = evaluator.evaluate()
        logger.info("Evaluation completed")
        logger.info("Evaluation metrics:")
        for metric_name, value in metrics.items():
            logger.info(f"{metric_name}: {value:.4f}")

if __name__ == '__main__':
    main()