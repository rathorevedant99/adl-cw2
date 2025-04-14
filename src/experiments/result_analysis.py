"""
Analyze and visualize results from weakly-supervised segmentation experiments.
Provides tools for comparing different models and visualizing segmentation outputs.
"""

import os
import torch
import yaml
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from tqdm import tqdm
import sys

# Add parent directory to path to import from src
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.segmentation_model import WeaklySupervisedSegmentationModel
from src.data import PetDataset
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F


class ExperimentAnalyzer:
    """Analyzes and visualizes results from weakly-supervised segmentation experiments."""
    
    def __init__(self, experiment_dir):
        """
        Initialize experiment analyzer.
        
        Args:
            experiment_dir: Directory with experiment results
        """
        self.experiment_dir = Path(experiment_dir)
        
        # Validate directory
        if not self.experiment_dir.exists():
            raise ValueError(f"Experiment directory {self.experiment_dir} does not exist")
        
        # Load experiment results
        self.results = self._load_results()
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 
                             'mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"Using device: {self.device}")
    
    def _load_results(self):
        """Load experiment results from JSON files."""
        results = {}
        
        # Find all result.json files
        result_files = list(self.experiment_dir.glob("**/results.json"))
        
        for result_file in result_files:
            # Get experiment name from parent directory
            experiment_name = result_file.parent.name
            
            with open(result_file, 'r') as f:
                results[experiment_name] = json.load(f)
        
        # Load ablation results if available
        ablation_file = self.experiment_dir / "ablation_results.json"
        if ablation_file.exists():
            with open(ablation_file, 'r') as f:
                results['ablation'] = json.load(f)
        
        return results
    
    def load_model(self, experiment_name):
        """
        Load a trained model from an experiment.
        
        Args:
            experiment_name: Name of the experiment
            
        Returns:
            Loaded model
        """
        model_path = self.experiment_dir / experiment_name / "model.pth"
        if not model_path.exists():
            raise ValueError(f"Model file not found at {model_path}")
        
        # Load configuration
        config_path = self.experiment_dir / experiment_name / "config.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            # Use default values
            config = {
                'model': {
                    'num_classes': 3,
                    'backbone': 'unet',
                    'cam_threshold': 0.2,
                    'region_growing_iterations': 5
                }
            }
        
        # Initialize model
        model = WeaklySupervisedSegmentationModel(
            num_classes=config['model']['num_classes'],
            backbone=config['model']['backbone'],
            cam_threshold=config['model'].get('cam_threshold', 0.2),
            region_growing_iterations=config['model'].get('region_growing_iterations', 5)
        ).to(self.device)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model
    
    def generate_comparison_table(self):
        """
        Generate a comparison table of all experiment results.
        
        Returns:
            DataFrame with comparison metrics
        """
        # Extract experiment names and metrics
        data = []
        
        for experiment_name, result in self.results.items():
            if experiment_name == 'ablation':
                continue
                
            # Get metrics
            metrics = result.get('metrics', {})
            
            # Create row
            row = {'Experiment': experiment_name}
            row.update(metrics)
            
            data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Save table
        output_path = self.experiment_dir / "comparison_table.csv"
        df.to_csv(output_path, index=False)
        
        print(f"Comparison table saved to {output_path}")
        
        return df
    
    def visualize_metrics(self, metric_name='miou'):
        """
        Visualize a specific metric across experiments.
        
        Args:
            metric_name: Name of the metric to visualize
            
        Returns:
            Path to saved plot
        """
        # Extract experiment names and metric values
        experiments = []
        values = []
        
        for experiment_name, result in self.results.items():
            if experiment_name == 'ablation':
                continue
                
            # Get metric value
            metrics = result.get('metrics', {})
            if metric_name in metrics:
                experiments.append(experiment_name)
                values.append(metrics[metric_name])
        
        # Create bar plot
        plt.figure(figsize=(12, 6))
        bars = plt.bar(experiments, values, alpha=0.7)
        
        # Add value labels
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{value:.4f}", ha='center', va='bottom', rotation=0)
        
        plt.grid(True, linestyle='--', alpha=0.5, axis='y')
        plt.title(f"Comparison of {metric_name.upper()} across Experiments")
        plt.xlabel("Experiment")
        plt.ylabel(f"{metric_name.upper()}")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save plot
        plots_dir = self.experiment_dir / "analysis_plots"
        plots_dir.mkdir(exist_ok=True)
        
        output_path = plots_dir / f"{metric_name}_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Metric visualization saved to {output_path}")
        
        return output_path
    
    def visualize_ablation_results(self, param_name):
        """
        Visualize results from an ablation study for a specific parameter.
        
        Args:
            param_name: Name of the parameter to visualize
            
        Returns:
            Path to saved plot
        """
        if 'ablation' not in self.results:
            print("No ablation results found")
            return None
        
        ablation_results = self.results['ablation']
        
        if param_name not in ablation_results:
            print(f"No ablation results found for parameter {param_name}")
            return None
        
        param_results = ablation_results[param_name]
        
        # Extract values and scores
        values = []
        scores = []
        
        # Determine main metric
        metric_name = 'miou'  # Default
        
        for value, result in param_results.items():
            if value == 'baseline':
                continue
                
            metrics = result.get('metrics', {})
            if metric_name in metrics:
                values.append(value)
                scores.append(metrics[metric_name])
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(values, scores, 'o-', linewidth=2, markersize=8)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.title(f"Ablation Study: Effect of {param_name} on {metric_name}")
        plt.xlabel(param_name)
        plt.ylabel(f"{metric_name.upper()}")
        
        # Highlight best value
        best_idx = np.argmax(scores)
        plt.scatter([values[best_idx]], [scores[best_idx]], color='red', s=100, 
                    zorder=10, label=f"Best: {values[best_idx]} ({scores[best_idx]:.4f})")
        plt.legend()
        
        # Save plot
        plots_dir = self.experiment_dir / "analysis_plots"
        plots_dir.mkdir(exist_ok=True)
        
        output_path = plots_dir / f"ablation_{param_name.replace('.', '_')}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Ablation visualization saved to {output_path}")
        
        return output_path
    
    def visualize_segmentation_examples(self, experiment_names, num_examples=5, dataset_root=None):
        """
        Visualize example segmentations from different experiments.
        
        Args:
            experiment_names: List of experiment names to compare
            num_examples: Number of examples to visualize
            dataset_root: Root directory of the dataset
            
        Returns:
            List of paths to saved visualizations
        """
        if dataset_root is None:
            print("Dataset root directory not provided, cannot visualize examples")
            return None
        
        # Initialize dataset
        dataset = PetDataset(
            root_dir=dataset_root,
            split='val',
            weak_supervision=False  # Use full supervision to get ground truth
        )
        
        # Create dataloaders with fixed random indices
        np.random.seed(42)
        indices = np.random.randint(0, len(dataset), size=num_examples)
        
        # Load models
        models = {}
        for name in experiment_names:
            try:
                models[name] = self.load_model(name)
            except Exception as e:
                print(f"Error loading model for experiment {name}: {e}")
        
        # Create visualization directory
        vis_dir = self.experiment_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        
        output_paths = []
        
        # Generate visualizations
        for i, idx in enumerate(indices):
            sample = dataset[idx]
            image = sample['image'].unsqueeze(0).to(self.device)
            mask = sample['mask']
            
            # Create figure
            n_cols = len(models) + 2  # Image + GT + models
            fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 4, 4))
            
            # Display original image
            img_np = sample['image'].permute(1, 2, 0).numpy()
            axes[0].imshow(img_np)
            axes[0].set_title("Original Image")
            axes[0].axis('off')
            
            # Display ground truth
            axes[1].imshow(mask.squeeze().numpy(), cmap='tab10', vmin=0, vmax=2)
            axes[1].set_title("Ground Truth")
            axes[1].axis('off')
            
            # Display predictions from each model
            for j, (name, model) in enumerate(models.items()):
                with torch.no_grad():
                    outputs = model(image, is_training=False)
                    seg_maps = outputs['segmentation_maps']
                    pred_mask = seg_maps.argmax(dim=1).squeeze().cpu().numpy()
                
                axes[j+2].imshow(pred_mask, cmap='tab10', vmin=0, vmax=2)
                axes[j+2].set_title(f"{name}")
                axes[j+2].axis('off')
            
            plt.tight_layout()
            
            # Save figure
            output_path = vis_dir / f"segmentation_example_{i+1}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            output_paths.append(output_path)
        
        print(f"Saved {len(output_paths)} segmentation visualizations to {vis_dir}")
        
        return output_paths
    
    def visualize_class_activation_maps(self, experiment_names, num_examples=5, dataset_root=None):
        """
        Visualize class activation maps from different experiments.
        
        Args:
            experiment_names: List of experiment names to compare
            num_examples: Number of examples to visualize
            dataset_root: Root directory of the dataset
            
        Returns:
            List of paths to saved visualizations
        """
        if dataset_root is None:
            print("Dataset root directory not provided, cannot visualize examples")
            return None
        
        # Initialize dataset
        dataset = PetDataset(
            root_dir=dataset_root,
            split='val',
            weak_supervision=True  # Use weak supervision to get class labels
        )
        
        # Create dataloaders with fixed random indices
        np.random.seed(42)
        indices = np.random.randint(0, len(dataset), size=num_examples)
        
        # Load models
        models = {}
        for name in experiment_names:
            try:
                models[name] = self.load_model(name)
            except Exception as e:
                print(f"Error loading model for experiment {name}: {e}")
        
        # Create visualization directory
        vis_dir = self.experiment_dir / "cam_visualizations"
        vis_dir.mkdir(exist_ok=True)
        
        output_paths = []
        
        # Generate visualizations
        for i, idx in enumerate(indices):
            sample = dataset[idx]
            image = sample['image'].unsqueeze(0).to(self.device)
            label = sample['label'].argmax().item()  # Get class index
            
            # Create figure
            n_cols = len(models) + 1  # Image + models
            fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 4, 4))
            
            # Display original image
            img_np = sample['image'].permute(1, 2, 0).numpy()
            axes[0].imshow(img_np)
            axes[0].set_title("Original Image")
            axes[0].axis('off')
            
            # Display CAMs from each model
            for j, (name, model) in enumerate(models.items()):
                with torch.no_grad():
                    # Get CAM for predicted class
                    cam = model.get_cam_maps(image, label)
                    cam_np = cam.squeeze().cpu().numpy()
                    
                    # Overlay CAM on image
                    axes[j+1].imshow(img_np)
                    im = axes[j+1].imshow(cam_np, cmap='jet', alpha=0.5)
                    axes[j+1].set_title(f"{name} CAM")
                    axes[j+1].axis('off')
            
            plt.tight_layout()
            
            # Save figure
            output_path = vis_dir / f"cam_example_{i+1}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            output_paths.append(output_path)
        
        print(f"Saved {len(output_paths)} CAM visualizations to {vis_dir}")
        
        return output_paths
    
    def visualize_learning_curves(self, experiment_names):
        """
        Visualize learning curves from different experiments.
        
        Args:
            experiment_names: List of experiment names to compare
            
        Returns:
            Path to saved plot
        """
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        for name in experiment_names:
            if name not in self.results:
                print(f"No results found for experiment {name}")
                continue
                
            result = self.results[name]
            history = result.get('history', {})
            
            # Extract training losses
            training_losses = history.get('training_losses', [])
            if training_losses:
                epochs = list(range(1, len(training_losses) + 1))
                ax1.plot(epochs, training_losses, '-o', label=name)
            
            # Extract validation metrics
            validation_metrics = history.get('validation_metrics', [])
            if validation_metrics and 'miou' in validation_metrics[0]:
                epochs = list(range(1, len(validation_metrics) + 1))
                val_mious = [metrics['miou'] for metrics in validation_metrics]
                ax2.plot(epochs, val_mious, '-o', label=name)
        
        # Configure plots
        ax1.set_title("Training Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()
        
        ax2.set_title("Validation mIoU")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("mIoU")
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()
        
        plt.tight_layout()
        
        # Save plot
        plots_dir = self.experiment_dir / "analysis_plots"
        plots_dir.mkdir(exist_ok=True)
        
        output_path = plots_dir / "learning_curves.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Learning curves saved to {output_path}")
        
        return output_path


def main():
    """Main function for analyzing experiment results."""
    parser = argparse.ArgumentParser(description='Analyze weakly-supervised segmentation experiments')
    parser.add_argument('--experiment_dir', type=str, required=True,
                     help='Directory with experiment results')
    parser.add_argument('--dataset_root', type=str, default=None,
                     help='Root directory of the dataset (for visualizations)')
    parser.add_argument('--visualize_examples', action='store_true',
                     help='Visualize segmentation examples')
    parser.add_argument('--visualize_cams', action='store_true',
                     help='Visualize class activation maps')
    parser.add_argument('--experiments', type=str, nargs='+', default=None,
                     help='Specific experiments to analyze (default: all)')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = ExperimentAnalyzer(args.experiment_dir)
    
    # Generate comparison table
    analyzer.generate_comparison_table()
    
    # Visualize metrics
    analyzer.visualize_metrics('miou')
    analyzer.visualize_metrics('pixel_accuracy')
    
    # Get experiment names
    if args.experiments:
        experiment_names = args.experiments
    else:
        experiment_names = [name for name in analyzer.results.keys() if name != 'ablation']
    
    # Visualize ablation results
    if 'ablation' in analyzer.results:
        ablation_params = [param for param in analyzer.results['ablation'].keys() if param != 'baseline']
        for param in ablation_params:
            analyzer.visualize_ablation_results(param)
    
    # Visualize learning curves
    analyzer.visualize_learning_curves(experiment_names)
    
    # Visualize segmentation examples
    if args.visualize_examples and args.dataset_root:
        analyzer.visualize_segmentation_examples(experiment_names, dataset_root=args.dataset_root)
    
    # Visualize class activation maps
    if args.visualize_cams and args.dataset_root:
        analyzer.visualize_class_activation_maps(experiment_names, dataset_root=args.dataset_root)


if __name__ == "__main__":
    main()
