#!/usr/bin/env python
import argparse
import os
import sys
import shutil
import logging
import datetime
import yaml
import torch
import optuna
from pathlib import Path
import torchvision.transforms as T
from torch.utils.data import ConcatDataset, Subset

from src.models.segmentation_model_resnet50 import (
    WeaklySupervisedSegmentationModelResNet50,
    FullySupervisedSegmentationModelResNet50
)
from src.models.segmentation_model_unet import (
    WeaklySupervisedSegmentationModelUNet,
    FullySupervisedSegmentationModelUNet
)
from src.data import PetDataset
from src.augment import AugmentedDataset
from src.training.trainer import Trainer
from src.training.evaluator import Evaluator

# Default transform pipeline
default_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def parse_args():
    parser = argparse.ArgumentParser(description='Optuna-tuned Semantic Segmentation')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'eval'],
                        help='Mode: train (with Optuna) or eval')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str,
                        help='Path to model checkpoint for evaluation')
    parser.add_argument('--download', action='store_true',
                        help='Download the dataset if missing')
    parser.add_argument('--n-trials', type=int, default=30,
                        help='Number of Optuna trials')
    parser.add_argument('--timeout', type=int, default=None,
                        help='Timeout in seconds for Optuna (only in train mode)')
    parser.add_argument('--subset-size', type=int, default=None,
                        help='If set, train on only this many random samples')
    return parser.parse_args()


def setup_logging(config):
    log_dir = Path(config['training']['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh = logging.FileHandler(log_dir / f"run_{datetime.datetime.now():%Y%m%d_%H%M%S}.log")
    fh.setFormatter(formatter)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in root.handlers[:]:
        root.removeHandler(h)
    root.addHandler(fh)
    root.addHandler(ch)
    return root


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def make_directories(config):
    Path(config['training']['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['training']['log_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['data']['root_dir']).mkdir(parents=True, exist_ok=True)


def download_and_prepare_dataset(config, logger, download_flag):
    root = config['data']['root_dir']
    images_dir = Path(root) / 'images'
    ann_dir = Path(root) / 'annotations'
    if download_flag or not images_dir.exists() or not ann_dir.exists() or len(list(images_dir.glob('*.jpg'))) == 0:
        logger.info("Downloading dataset...")
        PetDataset.download_dataset(root)
        logger.info("Download complete")
    trimaps = ann_dir / 'trimaps'
    if trimaps.exists() and not any(ann_dir.glob('*.png')):
        logger.info("Moving mask files from trimaps...")
        for f in trimaps.glob('*.png'):
            shutil.copy(f, ann_dir / f.name)
        logger.info("Mask files moved.")


def build_model(config, device):
    method = config['model'].get('method', 'WS').upper()
    backbone = config['model']['backbone']
    num_cls = config['model']['num_classes']
    if method == 'WS':
        if backbone == 'resnet50':
            return WeaklySupervisedSegmentationModelResNet50(num_cls).to(device)
        return WeaklySupervisedSegmentationModelUNet(num_cls).to(device)
    if backbone == 'resnet50':
        return FullySupervisedSegmentationModelResNet50(num_cls).to(device)
    return FullySupervisedSegmentationModelUNet(num_cls).to(device)


def objective(trial):
    # load and override config
    config = load_config(args.config)
    config['training']['learning_rate'] = trial.suggest_loguniform('lr', 1e-4, 1e-3)
    config['training']['weight_decay'] = trial.suggest_loguniform('wd', 1e-5, 1e-3)
    config['training']['seg_loss_weight'] = trial.suggest_float('wseg', 0.1, 0.5)
    config['training']['num_epochs'] = 3
    config['training']['save_interval'] = 1000

    # prepare dirs and logger
    make_directories(config)
    logger = setup_logging(config)
    logger.info(f"Trial #{trial.number}: lr={config['training']['learning_rate']}, "
                + f"wd={config['training']['weight_decay']}, wseg={config['training']['seg_loss_weight']}")

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataset
    download_and_prepare_dataset(config, logger, args.download)
    is_weak = config['model']['method'].upper() == 'WS'
    train_ds = PetDataset(config['data']['root_dir'], split='train', weak_supervision=is_weak, transform=default_transform)
    logger.info(f"Raw train set size: {len(train_ds)} samples")
    # apply subset if requested
    if args.subset_size:
        N = len(train_ds)
        k = min(args.subset_size, N)
        idx = torch.randperm(N)[:k].tolist()
        train_ds = Subset(train_ds, idx)
        logger.info(f"Subsampled train set to {len(train_ds)} samples")

    val_ds = PetDataset(config['data']['root_dir'], split='val', weak_supervision=is_weak, transform=default_transform)
    logger.info(f"Validation set size: {len(val_ds)} samples")

    # augmentation for WS
    if is_weak:
        aug = AugmentedDataset(train_ds)
        aug._build_augmented_indices()
        full_train = ConcatDataset([train_ds, aug])
        logger.info(f"Combined WS train size: {len(full_train)} (subset + augmented)")
    else:
        full_train = train_ds

    # model & trainer
    model = build_model(config, device)
    trainer = Trainer(model, full_train, val_ds, config, optuna_trial=trial)

    # train
    try:
        trainer.train()
    except optuna.TrialPruned:
        raise

    # return final val loss
    return trainer.epoch_val_losses[-1]


def main():
    global args
    args = parse_args()
    if args.mode == 'train':
        study = optuna.create_study(storage = "sqlite:///optuna_pets.db", study_name="pet_segmentations_ws", direction='minimize', sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner(n_warmup_steps=2))
        study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)
        print("Best hyperparameters:", study.best_trial.params)
        import joblib
        joblib.dump(study, "experiments/optuna_study.pkl")
    else:
        # evaluation path remains unchanged
        config = load_config(args.config)
        make_directories(config)
        logger = setup_logging(config)
        download_and_prepare_dataset(config, logger, args.download)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        test_ds = PetDataset(config['data']['root_dir'], split='test', weak_supervision=False, transform=default_transform)
        logger.info(f"Test set size: {len(test_ds)} samples")
        model = build_model(config, device)
        if not args.checkpoint:
            raise ValueError("Please provide --checkpoint for eval mode")
        ckpt_path = Path(config['evaluation']['checkpoint_dir']) / args.checkpoint
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        evaluator = Evaluator(model, test_ds, config)
        metrics = evaluator.evaluate()
        print("Evaluation metrics:", metrics)

if __name__ == '__main__':
    main()
