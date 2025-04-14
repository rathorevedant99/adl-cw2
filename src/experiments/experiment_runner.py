"""
Experiment runner for weakly-supervised segmentation framework.
Runs multiple experiment configurations and saves results.
"""

import os
import torch
import yaml
import json
from pathlib import Path
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from src.models.segmentation_model import WeaklySupervisedSegmentationModel
from src.data import PetDataset
from src.training.weakly_supervised_trainer import WeaklySupervisedTrainer
from src.training.trainer import Trainer
from src.training.evaluator import Evaluator

# Get logger
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Runs weakly-supervised segmentation experiments with different configurations."""
    
    def __init__(self, config_path, experiment_dir=None):
        """
        Initialize experiment runner.
        
        Args:
            config_path: Path to base configuration file
            experiment_dir: Directory to save experiment results
        """
        self.base_config = self._load_config(config_path)
        
        # Setup experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = Path(experiment_dir or f"experiments/results_{timestamp}")
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 
                             'mps' if torch.backends.mps.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Results storage
        self.results = {}
    
    def _load_config(self, config_path):
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self):
        """Setup logging for experiments."""
        log_file = self.experiment_dir / "experiments.log"
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
    
    def run_fully_supervised_baseline(self, data_split=0.8):
        """
        Run fully-supervised baseline model for comparison.
        
        Args:
            data_split: Proportion of data to use for training
        
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("=== Running Fully-Supervised Baseline ===")
        
        # Initialize dataset with full supervision
        dataset = PetDataset(
            root_dir=self.base_config['data']['root_dir'],
            split='train',
            weak_supervision=False  # Use full supervision (pixel-level masks)
        )
        
        # Create train/val split
        train_size = int(len(dataset) * data_split)
        val_size = len(dataset) - train_size
        
        from torch.utils.data import random_split
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size], 
            generator=torch.Generator().manual_seed(42)
        )
        
        logger.info(f"Training on {train_size} samples, validating on {val_size} samples")
        
        # Initialize model
        model = WeaklySupervisedSegmentationModel(
            num_classes=self.base_config['model']['num_classes'],
            backbone=self.base_config['model']['backbone']
        ).to(self.device)
        
        # Setup training config
        training_config = {
            'device': str(self.device),
            'num_epochs': self.base_config['training']['num_epochs'],
            'batch_size': self.base_config['training']['batch_size'],
            'learning_rate': self.base_config['training']['learning_rate'],
            'checkpoint_dir': str(self.experiment_dir / "fully_supervised" / "checkpoints"),
            'log_dir': str(self.experiment_dir / "fully_supervised" / "logs")
        }
        
        # Create checkpoint directory
        Path(training_config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
        Path(training_config['log_dir']).mkdir(parents=True, exist_ok=True)
        
        # Train model
        trainer = Trainer(
            model=model,
            dataset=train_dataset,
            config={
                'training': training_config,
                'model': self.base_config['model']
            }
        )
        
        logger.info("Training fully-supervised model...")
        trainer.train()
        
        # Evaluate model
        evaluator = Evaluator(
            model=model,
            dataset=val_dataset,
            config={
                'evaluation': {
                    'batch_size': self.base_config['evaluation']['batch_size']
                },
                'model': self.base_config['model']
            }
        )
        
        logger.info("Evaluating fully-supervised model...")
        metrics = evaluator.evaluate()
        
        # Save results
        result_path = self.experiment_dir / "fully_supervised" / "results.json"
        with open(result_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save model
        model_path = self.experiment_dir / "fully_supervised" / "model.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'metrics': metrics
        }, model_path)
        
        logger.info(f"Fully-supervised baseline results: {metrics}")
        
        # Store results
        self.results['fully_supervised'] = {
            'metrics': metrics,
            'model_path': str(model_path),
            'config': {
                'data_split': data_split,
                'weak_supervision': False
            }
        }
        
        return metrics
    
    def run_weakly_supervised_experiment(self, experiment_name, config_overrides=None, 
                                        data_split=0.8, use_additional_data=False,
                                        additional_data_dir=None):
        """
        Run weakly-supervised experiment with specified configuration.
        
        Args:
            experiment_name: Name of the experiment
            config_overrides: Dictionary with configuration overrides
            data_split: Proportion of data to use for training
            use_additional_data: Whether to use additional weakly-labeled data
            additional_data_dir: Directory with additional data
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"=== Running Weakly-Supervised Experiment: {experiment_name} ===")
        
        # Create experiment config by applying overrides
        experiment_config = self._create_experiment_config(config_overrides)
        
        # Initialize dataset with weak supervision
        dataset = PetDataset(
            root_dir=self.base_config['data']['root_dir'],
            split='train',
            weak_supervision=True  # Use weak supervision (image-level labels)
        )
        
        # Create train/val split
        train_size = int(len(dataset) * data_split)
        val_size = len(dataset) - train_size
        
        from torch.utils.data import random_split
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size], 
            generator=torch.Generator().manual_seed(42)
        )
        
        logger.info(f"Training on {train_size} samples, validating on {val_size} samples")
        
        # Initialize model
        model = WeaklySupervisedSegmentationModel(
            num_classes=experiment_config['model']['num_classes'],
            backbone=experiment_config['model']['backbone'],
            cam_threshold=experiment_config['model'].get('cam_threshold', 0.2),
            region_growing_iterations=experiment_config['model'].get('region_growing_iterations', 5)
        ).to(self.device)
        
        # Setup training config
        experiment_dir = self.experiment_dir / experiment_name
        training_config = {
            'device': str(self.device),
            'num_epochs': experiment_config['training']['num_epochs'],
            'batch_size': experiment_config['training']['batch_size'],
            'learning_rate': experiment_config['training']['learning_rate'],
            'weight_decay': experiment_config['training'].get('weight_decay', 1e-5),
            'checkpoint_dir': str(experiment_dir / "checkpoints"),
            'log_dir': str(experiment_dir / "logs"),
            'num_classes': experiment_config['model']['num_classes'],
            'generate_pseudo_labels': experiment_config['training'].get('generate_pseudo_labels', True),
            'consistency_weight': experiment_config['training'].get('consistency_weight', 1.0),
            'curriculum_learning': experiment_config['training'].get('curriculum_learning', True),
            'strong_augment': experiment_config['training'].get('strong_augment', True),
            'save_frequency': experiment_config['training'].get('save_frequency', 5)
        }
        
        # Create checkpoint directory
        Path(training_config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
        Path(training_config['log_dir']).mkdir(parents=True, exist_ok=True)
        
        # Check for additional data
        if use_additional_data and additional_data_dir:
            additional_data_path = Path(additional_data_dir)
            if not additional_data_path.exists() or not any(additional_data_path.iterdir()):
                logger.warning(f"Additional data directory {additional_data_path} does not exist or is empty.")
                additional_data_path = None
            else:
                logger.info(f"Using additional data from {additional_data_path}")
        else:
            additional_data_path = None
        
        # Train model
        trainer = WeaklySupervisedTrainer(
            model=model,
            original_dataset=train_dataset,
            additional_data_dir=additional_data_path,
            config=training_config
        )
        
        logger.info(f"Training weakly-supervised model for experiment '{experiment_name}'...")
        history = trainer.train()
        
        # Evaluate model
        # Use a separate dataset with full supervision for evaluation
        eval_dataset = PetDataset(
            root_dir=self.base_config['data']['root_dir'],
            split='val',
            weak_supervision=False  # Use full supervision for evaluation
        )
        
        evaluator = Evaluator(
            model=model,
            dataset=eval_dataset,
            config={
                'evaluation': {
                    'batch_size': self.base_config['evaluation']['batch_size']
                },
                'model': self.base_config['model']
            }
        )
        
        logger.info(f"Evaluating weakly-supervised model for experiment '{experiment_name}'...")
        metrics = evaluator.evaluate()
        
        # Save results
        result_path = experiment_dir / "results.json"
        with open(result_path, 'w') as f:
            json.dump({
                'metrics': metrics,
                'config_overrides': config_overrides,
                'history': {
                    'training_losses': [item['loss'] for item in history.get('training_losses', [])],
                    'validation_metrics': [item['metrics'] for item in history.get('validation_metrics', [])]
                }
            }, f, indent=2)
        
        # Save model
        model_path = experiment_dir / "model.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'history': history
        }, model_path)
        
        # Save configuration
        config_path = experiment_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(experiment_config, f)
        
        logger.info(f"Experiment '{experiment_name}' results: {metrics}")
        
        # Store results
        self.results[experiment_name] = {
            'metrics': metrics,
            'model_path': str(model_path),
            'config': {
                'overrides': config_overrides,
                'data_split': data_split,
                'weak_supervision': True,
                'use_additional_data': use_additional_data
            },
            'history': history
        }
        
        return metrics
    
    def _create_experiment_config(self, overrides=None):
        """Create experiment configuration by applying overrides to base config."""
        import copy
        
        # Create a deep copy of the base config
        config = copy.deepcopy(self.base_config)
        
        # Apply overrides
        if overrides:
            for key, value in overrides.items():
                # Handle nested keys like 'model.cam_threshold'
                keys = key.split('.')
                target = config
                
                # Navigate to the target dictionary
                for k in keys[:-1]:
                    if k not in target:
                        target[k] = {}
                    target = target[k]
                
                # Set the value
                target[keys[-1]] = value
        
        return config
    
    def run_ablation_study(self, base_experiment_name='baseline_weak', parameters=None):
        """
        Run ablation study by varying important hyperparameters.
        
        Args:
            base_experiment_name: Name of the base experiment
            parameters: Dictionary mapping parameter names to lists of values to try
            
        Returns:
            Dictionary with results for each parameter configuration
        """
        logger.info("=== Running Ablation Study ===")
        
        # Default parameters to ablate if none provided
        if parameters is None:
            parameters = {
                'model.cam_threshold': [0.1, 0.2, 0.3, 0.4],
                'model.region_growing_iterations': [3, 5, 7, 10],
                'training.consistency_weight': [0.5, 1.0, 2.0],
                'training.generate_pseudo_labels': [True, False]
            }
        
        # Store ablation results
        ablation_results = {}
        
        # Run baseline experiment first
        baseline_metrics = self.run_weakly_supervised_experiment(
            base_experiment_name,
            config_overrides=None  # Use default config
        )
        
        ablation_results['baseline'] = {
            'metrics': baseline_metrics,
            'config_overrides': None
        }
        
        # Run experiment for each parameter configuration
        for param_name, param_values in parameters.items():
            param_results = {}
            
            for value in param_values:
                # Skip baseline value if it matches
                if self._is_baseline_value(param_name, value):
                    param_results[str(value)] = ablation_results['baseline']
                    continue
                
                # Create experiment name
                exp_name = f"{base_experiment_name}_{param_name.replace('.', '_')}_{value}"
                
                # Create config override
                config_override = {param_name: value}
                
                # Run experiment
                logger.info(f"Running ablation for {param_name}={value}")
                metrics = self.run_weakly_supervised_experiment(
                    exp_name,
                    config_overrides=config_override
                )
                
                # Store results
                param_results[str(value)] = {
                    'metrics': metrics,
                    'config_overrides': config_override
                }
            
            ablation_results[param_name] = param_results
        
        # Save overall ablation results
        result_path = self.experiment_dir / "ablation_results.json"
        with open(result_path, 'w') as f:
            json.dump(ablation_results, f, indent=2)
        
        # Generate ablation plots
        self._generate_ablation_plots(ablation_results)
        
        logger.info("Ablation study completed")
        
        return ablation_results
    
    def _is_baseline_value(self, param_name, value):
        """Check if a parameter value matches the baseline value."""
        # Navigate to the parameter in the base config
        keys = param_name.split('.')
        target = self.base_config
        
        for key in keys:
            if key not in target:
                return False
            target = target[key]
        
        return target == value
    
    def _generate_ablation_plots(self, ablation_results):
        """Generate plots visualizing ablation study results."""
        plots_dir = self.experiment_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Extract primary metric (e.g., mIoU)
        metric_name = 'miou'  # Default to mIoU
        
        # For each ablated parameter
        for param_name, param_results in ablation_results.items():
            if param_name == 'baseline':
                continue
                
            # Extract values and scores
            values = []
            scores = []
            
            for value, result in param_results.items():
                values.append(value)
                scores.append(result['metrics'].get(metric_name, 0))
            
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
            plot_path = plots_dir / f"ablation_{param_name.replace('.', '_')}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    def compare_all_experiments(self):
        """Compare results across all experiments and generate comparison plots."""
        logger.info("=== Comparing All Experiments ===")
        
        # Extract metrics from all experiments
        experiment_names = []
        metrics = {}
        
        for name, result in self.results.items():
            experiment_names.append(name)
            
            for metric_name, value in result['metrics'].items():
                if metric_name not in metrics:
                    metrics[metric_name] = []
                metrics[metric_name].append(value)
        
        # Create comparison plots
        plots_dir = self.experiment_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        for metric_name, values in metrics.items():
            plt.figure(figsize=(12, 6))
            
            # Create bar chart
            bars = plt.bar(experiment_names, values, alpha=0.7)
            
            # Add value labels on top of bars
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
            plot_path = plots_dir / f"comparison_{metric_name}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        # Create summary table
        summary = {
            'Experiment': experiment_names
        }
        
        for metric_name in metrics:
            summary[metric_name.upper()] = metrics[metric_name]
        
        summary_df = pd.DataFrame(summary)
        
        # Save summary table
        summary_path = self.experiment_dir / "experiment_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        
        # Generate HTML report
        self._generate_html_report(summary_df, metrics)
        
        logger.info(f"Comparison completed. Summary saved to {summary_path}")
        
        return summary_df
    
    def _generate_html_report(self, summary_df, metrics):
        """Generate HTML report with experiment results."""
        html_path = self.experiment_dir / "experiment_report.html"
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Weakly-Supervised Segmentation Experiments</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ padding: 8px; text-align: left; border: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .plot-container {{ margin-bottom: 30px; }}
                .highlight {{ font-weight: bold; color: #009900; }}
            </style>
        </head>
        <body>
            <h1>Weakly-Supervised Segmentation Experiment Results</h1>
            <p>Experiment run on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <h2>Summary of Results</h2>
            <table>
                <tr>
                    {"".join(f"<th>{col}</th>" for col in summary_df.columns)}
                </tr>
                {"".join(f"<tr>{''.join(f'<td>{cell}</td>' for cell in row)}</tr>" for row in summary_df.values)}
            </table>
            
            <h2>Experiment Plots</h2>
        """
        
        # Add plot images
        plots_dir = self.experiment_dir / "plots"
        if plots_dir.exists():
            plot_files = list(plots_dir.glob("*.png"))
            
            for plot_file in plot_files:
                relative_path = plot_file.relative_to(self.experiment_dir)
                html_content += f"""
                <div class="plot-container">
                    <h3>{plot_file.stem.replace('_', ' ').title()}</h3>
                    <img src="{relative_path}" alt="{plot_file.stem}" style="max-width: 800px;">
                </div>
                """
        
        # Add experiment details
        html_content += f"""
            <h2>Experiment Details</h2>
        """
        
        for name, result in self.results.items():
            html_content += f"""
            <h3>Experiment: {name}</h3>
            <table>
                <tr><th>Parameter</th><th>Value</th></tr>
                <tr><td>Weak Supervision</td><td>{result['config']['weak_supervision']}</td></tr>
                <tr><td>Data Split</td><td>{result['config']['data_split']}</td></tr>
                {"".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in result.get('config', {}).get('overrides', {}).items() if k != 'weak_supervision' and k != 'data_split')}
            </table>
            
            <h4>Metrics</h4>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                {"".join(f"<tr><td>{k}</td><td>{v:.4f}</td></tr>" for k, v in result['metrics'].items())}
            </table>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        # Write HTML file
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML report generated at {html_path}")


def setup_experiment_configs():
    """Define experiment configurations for comprehensive comparison."""
    experiment_configs = {
        # Baseline weakly-supervised with default parameters
        'baseline_weak': None,
        
        # Varying CAM threshold
        'cam_thresh_low': {'model.cam_threshold': 0.1},
        'cam_thresh_high': {'model.cam_threshold': 0.3},
        
        # Varying region growing iterations
        'region_growing_low': {'model.region_growing_iterations': 3},
        'region_growing_high': {'model.region_growing_iterations': 10},
        
        # With and without consistency loss
        'no_consistency': {'training.consistency_weight': 0.0},
        'high_consistency': {'training.consistency_weight': 2.0},
        
        # With and without pseudo-labeling
        'no_pseudo_labels': {'training.generate_pseudo_labels': False},
        
        # With and without curriculum learning
        'no_curriculum': {'training.curriculum_learning': False},
        
        # With varying learning rates
        'low_lr': {'training.learning_rate': 1e-5},
        'high_lr': {'training.learning_rate': 1e-3}
    }
    
    return experiment_configs


# Main function has been removed and integrated into the unified run.py CLI
