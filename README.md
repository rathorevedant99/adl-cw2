# Weakly-Supervised Semantic Segmentation for Pet Images

This project implements a weakly-supervised neural network for semantic segmentation of pet images using the Oxford-IIIT Pet dataset. The implementation focuses on using image-level labels and/or bounding box annotations as weak supervision signals for training.

## Project Structure

```
.
├── data/               # Dataset storage and processing
├── src/               # Source code
│   ├── models/       # Neural network architectures
│   ├── data/         # Data loading and preprocessing
│   ├── training/     # Training loops and utilities
│   └── utils/        # Helper functions
├── experiments/      # Experiment configurations and results
├── requirements.txt  # Project dependencies
└── main.py          # Main entry point
```

## Setup

1. Create a virtual environment:
```bash
python -m venv .env
source .env/bin/activate  # On Unix/macOS
# or
.env\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the Oxford-IIIT Pet dataset:
```bash
python src/data/download_dataset.py
```

## Usage

1. Train the model:
```bash
python main.py --mode train --config experiments/config.yaml
```

2. Evaluate the model:
```bash
python main.py --mode evaluate --checkpoint path/to/checkpoint.pth
```

## Features

- Weakly-supervised semantic segmentation using image-level labels
- Baseline comparison with fully-supervised methods
- Ablation studies for different configurations
- Comprehensive evaluation metrics

## Requirements

- Python 3.8+
- PyTorch 2.6.0
- torchvision 0.21.0
- Other dependencies listed in requirements.txt

## License

This project is licensed under the terms of the LICENSE file in the root directory.