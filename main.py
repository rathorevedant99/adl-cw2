import argparse
import yaml
import torch
from pathlib import Path

from src.models.segmentation_model import WeaklySupervisedSegmentationModel
from src.data.dataset import PetDataset
from src.training.trainer import Trainer
from src.training.evaluator import Evaluator

def parse_args():
    parser = argparse.ArgumentParser(description='Weakly-Supervised Semantic Segmentation')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'evaluate'],
                      help='Mode to run the model in')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str,
                      help='Path to model checkpoint for evaluation')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize dataset
    dataset = PetDataset(
        root_dir=config['data']['root_dir'],
        split='train' if args.mode == 'train' else 'val',
        weak_supervision=True
    )
    
    # Initialize model
    model = WeaklySupervisedSegmentationModel(
        num_classes=config['model']['num_classes'],
        backbone=config['model']['backbone']
    ).to(device)
    
    if args.mode == 'train':
        trainer = Trainer(
            model=model,
            dataset=dataset,
            config=config['training']
        )
        trainer.train()
    else:
        if not args.checkpoint:
            raise ValueError("Checkpoint path must be provided for evaluation mode")
        
        # Load checkpoint
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        evaluator = Evaluator(
            model=model,
            dataset=dataset,
            config=config['evaluation']
        )
        metrics = evaluator.evaluate()
        print("Evaluation metrics:", metrics)

if __name__ == '__main__':
    main()
