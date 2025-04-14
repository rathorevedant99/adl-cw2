# Weakly-Supervised Semantic Segmentation for Pet Images

This project implements a weakly-supervised neural network for semantic segmentation of pet images using the Oxford-IIIT Pet dataset. The implementation focuses on using image-level labels and/or bounding box annotations as weak supervision signals for training.

## Overview

The project implements a weakly-supervised segmentation approach using Class Activation Maps (CAM) and Region Growing for generating pseudo-masks. It supports both fully-supervised and weakly-supervised training, as well as integration of additional weakly-labeled data from external sources.

### Key Features

- **Weakly-supervised segmentation**: Train with image-level labels only
- **Class Activation Mapping (CAM)**: Generate attention maps from classification networks
- **Region Growing**: Refine initial attention maps into more precise segmentation masks
- **Data collection interface**: Collect additional weakly-labeled data from external sources
- **Comprehensive experiment framework**: Run ablation studies and parameter sweeps

## Project Structure

```
.
├── data/                 # Dataset storage
│   ├── oxford_pet/       # Main dataset
│   └── additional/       # Additional weakly-labeled data
├── experiments/          # Experiment results, checkpoints and logs
├── src/                  # Source code
│   ├── models/           # Neural network architectures
│   │   └── segmentation_model.py  # Main segmentation model
│   ├── training/         # Training loops and utilities
│   │   ├── trainer.py    # Fully-supervised trainer
│   │   ├── weakly_supervised_trainer.py  # Weakly-supervised trainer
│   │   └── evaluator.py  # Model evaluation
│   ├── experiments/      # Experiment utilities
│   │   ├── experiment_runner.py   # Runs multiple experiments
│   │   ├── result_analysis.py     # Analyzes experiment results
│   │   └── generate_report.py     # Generates experiment reports
│   ├── data.py           # Data loading and preprocessing
│   └── data_collection.py  # External data collection utilities
├── requirements.txt      # Project dependencies
├── run.py               # Unified entry point
├── config.yaml          # Configuration file
└── README.md            # Project documentation
```

## Setup

1. Create and activate virtual environment:
```bash
python3.10 -m venv .env
source .env/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage Guide

This project provides a comprehensive framework for weakly-supervised semantic segmentation with a unified command-line interface through `run.py`.

### Configuration

The system uses a YAML configuration file to control all aspects of training, evaluation, and experimentation:

```bash
# View or edit the configuration file
vim config.yaml
```

You can override any config parameter via command-line arguments.

### Dataset Setup

```bash
# Download the Oxford-IIIT Pet dataset automatically
python run.py --mode train --download --config config.yaml

# If you have the dataset already, specify its location
python run.py --mode train --config config.yaml
```

### Training Commands

#### Basic Training

```bash
# Fully-supervised training (with pixel-level annotations)
python run.py --mode train --config config.yaml --device cuda --batch_size 32 --num_epochs 30

# Weakly-supervised training (with image-level labels only)
python run.py --mode train_weak --config config.yaml --device cuda --batch_size 32 --num_epochs 50
```

#### Training with Specific Parameters

```bash
# Adjusting learning rate and optimization parameters
python run.py --mode train_weak --config config.yaml --learning_rate 0.0005 --weight_decay 0.00005

# Configuring model parameters
python run.py --mode train_weak --config config.yaml --cam_threshold 0.3 --region_growing_iterations 7

# Adjusting loss weights
python run.py --mode train_weak --config config.yaml --consistency_weight 1.5
```

#### Training with Additional Data

```bash
# Use additional weakly-labeled data
python run.py --mode train_weak --config config.yaml --use_additional_data --additional_data_dir data/additional

# Continue training from a checkpoint
python run.py --mode train_weak --config config.yaml --checkpoint checkpoints/checkpoint_epoch_20.pth
```

### Evaluation Commands

```bash
# Basic evaluation on test set
python run.py --mode eval --checkpoint final_model_weak.pth --eval_split test

# Evaluation with visualization
python run.py --mode eval --checkpoint final_model_weak.pth --eval_split test --visualize --save_predictions

# Evaluation on validation set
python run.py --mode eval --checkpoint final_model_weak.pth --eval_split val
```

### Data Collection Commands

```bash
# Collect data from Flickr
python run.py --mode collect_data --flickr_key YOUR_KEY --flickr_secret YOUR_SECRET --max_per_source 200

# Collect data from multiple sources
python run.py --mode collect_data --flickr_key YOUR_KEY --flickr_secret YOUR_SECRET \
                               --petfinder_key YOUR_KEY --petfinder_secret YOUR_SECRET

# Specify output directory for collected data
python run.py --mode collect_data --flickr_key YOUR_KEY --additional_data_dir custom/data/path
```

### Experiment Commands

```bash
# Run baseline experiments
python run.py --mode experiment --config config.yaml --run_baseline

# Run ablation studies
python run.py --mode experiment --config config.yaml --run_ablation

# Run comprehensive experiments
python run.py --mode experiment --config config.yaml --run_baseline --run_ablation \
                               --experiment_dir experiments/results_$(date +%Y%m%d)
```

### End-to-End Workflow Example

Here's a complete workflow from data collection to evaluation:

```bash
# 1. Collect additional data
python run.py --mode collect_data --flickr_key YOUR_KEY --flickr_secret YOUR_SECRET

# 2. Train weakly-supervised model with additional data
python run.py --mode train_weak --config config.yaml --use_additional_data \
                              --num_epochs 50 --batch_size 32 --device cuda

# 3. Evaluate the trained model
python run.py --mode eval --checkpoint final_model_weak.pth --eval_split test --visualize

# 4. Run experiments to compare different configurations
python run.py --mode experiment --config config.yaml --run_ablation
```

## Command-Line Options

The framework supports numerous command-line options. View all available options with:

```bash
python run.py --help
```

### Common Options

| Option | Description | Default |
|--------|-------------|--------|
| `--mode` | Operation mode (train, train_weak, eval, collect_data, experiment) | Required |
| `--config` | Path to configuration file | config.yaml |
| `--device` | Computing device (cuda, mps, cpu) | From config |
| `--checkpoint` | Path to model checkpoint for evaluation/continued training | None |
| `--download` | Download the Oxford-IIIT Pet dataset if not present | False |
| `--output_dir` | Directory to save outputs | From config |
| `--batch_size` | Batch size for training/evaluation | From config |
| `--num_epochs` | Number of training epochs | From config |
| `--learning_rate` | Learning rate | From config |

### Model-Specific Options

| Option | Description | Default |
|--------|-------------|--------|
| `--base_filters` | Number of base filters in U-Net | From config |
| `--cam_threshold` | Threshold for class activation maps | From config |
| `--region_growing_iterations` | Number of region growing iterations | From config |

### Weakly-Supervised Options

| Option | Description | Default |
|--------|-------------|--------|
| `--use_additional_data` | Use additional weakly-labeled data | False |
| `--additional_data_dir` | Path to additional data | data/additional |
| `--consistency_weight` | Weight for consistency loss | From config |
| `--curriculum_learning` | Enable curriculum learning | From config |
| `--generate_pseudo_labels` | Generate pseudo labels during training | From config |

### Evaluation Options

| Option | Description | Default |
|--------|-------------|--------|
| `--eval_split` | Dataset split to evaluate on (train, val, test) | test |
| `--save_predictions` | Save model predictions | False |
| `--visualize` | Visualize model predictions | False |

## License

This project is licensed under the terms of the LICENSE file in the root directory.