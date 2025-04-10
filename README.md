# Weakly-Supervised Semantic Segmentation for Pet Images

This project implements a weakly-supervised neural network for semantic segmentation of pet images using the Oxford-IIIT Pet dataset. The implementation focuses on using image-level labels and/or bounding box annotations as weak supervision signals for training.

## Docs
- kaiming_normal_: https://paperswithcode.com/method/he-initialization
- Global Average Pooling: https://paperswithcode.com/method/global-average-pooling
- Class Activation Mapping: https://zilliz.com/learn/class-activation-mapping-CAM
- GradCAM: https://medium.com/@bmuskan007/grad-cam-a-beginners-guide-adf68e80f4bb

## Project Structure

```
.
├── data/             # Dataset storage and processing
├── src/              # Source code
│   ├── models/       # Neural network architectures
│   ├── data.py       # Data loading and preprocessing
│   ├── training/     # Training loops and utilities
│   └── utils/        # Helper functions
├── experiments/      # Experiment configurations and results
├── requirements.txt  # Project dependencies
└── main.py           # Main entry point
```

## Setup

1. Create a virtual environment:
```bash
python3.10 -m venv .env
source .env/bin/activate  # On Unix/macOS
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Train the model:
```bash
python main.py --mode train --config config.yaml
```

2. Evaluate the model:
```bash
python main.py --mode evaluate --checkpoint path/to/checkpoint.pth
```

## License

This project is licensed under the terms of the LICENSE file in the root directory.