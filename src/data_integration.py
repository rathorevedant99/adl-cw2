"""
Integration of additional weakly-labeled and unlabeled data for weakly-supervised learning.
Implements pseudo-labeling, consistency regularization, and curriculum learning strategies.
"""

import os
import torch
import numpy as np
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image
import json
import random
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdditionalDataset(Dataset):
    """Dataset for additional weakly-labeled or unlabeled pet images."""
    
    def __init__(self, data_dir, transform=None, label_type='weak', confidence_threshold=0.8):
        """
        Initialize dataset.
        
        Args:
            data_dir: Directory containing the images
            transform: Image transformations
            label_type: Type of labels ('weak', 'pseudo', 'none')
            confidence_threshold: Threshold for confidence in pseudo-labels
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.label_type = label_type
        self.confidence_threshold = confidence_threshold
        
        # Get all image files
        self.image_paths = []
        self.source_types = []
        
        for source in ['flickr', 'petfinder', 'coco', 'openimages']:
            source_dir = self.data_dir / source
            if source_dir.exists():
                img_paths = list(source_dir.glob('*.jpg'))
                self.image_paths.extend(img_paths)
                self.source_types.extend([source] * len(img_paths))
        
        logger.info(f"Found {len(self.image_paths)} additional images for {label_type} labels")
        
        # Load metadata and pseudo-labels if available
        self.metadata = {}
        self.pseudo_labels = {}
        self.pseudo_mask_paths = {}
        
        for img_path in self.image_paths:
            # Check for metadata
            meta_path = img_path.with_name(f"{img_path.stem}_meta.json")
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    self.metadata[str(img_path)] = json.load(f)
            
            # Check for pseudo-labels
            if label_type == 'pseudo':
                pseudo_label_path = img_path.with_name(f"{img_path.stem}_pseudo.json")
                if pseudo_label_path.exists():
                    with open(pseudo_label_path, 'r') as f:
                        pseudo_data = json.load(f)
                        if pseudo_data.get('confidence', 0) >= self.confidence_threshold:
                            self.pseudo_labels[str(img_path)] = pseudo_data
                            
                            # Check for pseudo-mask
                            mask_path = img_path.with_name(f"{img_path.stem}_pseudo_mask.png")
                            if mask_path.exists():
                                self.pseudo_mask_paths[str(img_path)] = mask_path
        
        # If using pseudo-labels, only keep images with valid labels
        if label_type == 'pseudo':
            valid_paths = [p for p in self.image_paths if str(p) in self.pseudo_labels]
            valid_sources = [self.source_types[i] for i, p in enumerate(self.image_paths) if str(p) in self.pseudo_labels]
            
            self.image_paths = valid_paths
            self.source_types = valid_sources
            
            logger.info(f"Using {len(self.image_paths)} images with pseudo-labels above confidence threshold")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = str(self.image_paths[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        sample = {'image': image, 'path': img_path, 'source': self.source_types[idx]}
        
        # Add labels based on type
        if self.label_type == 'weak':
            # Try to extract weak labels from metadata
            label = self._get_weak_label(img_path)
            sample['label'] = label
            sample['has_label'] = True
            
        elif self.label_type == 'pseudo':
            # Add pseudo-labels and masks
            if img_path in self.pseudo_labels:
                pseudo_data = self.pseudo_labels[img_path]
                sample['label'] = torch.tensor(pseudo_data['label'], dtype=torch.long)
                sample['confidence'] = torch.tensor(pseudo_data['confidence'])
                sample['has_label'] = True
                
                if img_path in self.pseudo_mask_paths:
                    mask_path = self.pseudo_mask_paths[img_path]
                    mask = Image.open(mask_path)
                    
                    # Transform mask if needed
                    if self.transform:
                        # Need special handling for mask transforms
                        mask_transform = transforms.Compose([
                            transforms.Resize(image.shape[-2:]),
                            transforms.ToTensor()
                        ])
                        mask = mask_transform(mask)
                    
                    sample['mask'] = mask
                    sample['has_mask'] = True
                else:
                    sample['has_mask'] = False
            else:
                sample['has_label'] = False
                sample['has_mask'] = False
        else:
            # Unlabeled
            sample['has_label'] = False
            sample['has_mask'] = False
            
        return sample
    
    def _get_weak_label(self, img_path):
        """
        Extract weak labels from metadata.
        For now, just determine if it's a cat or dog.
        """
        metadata = self.metadata.get(img_path, {})
        source = self.source_types[self.image_paths.index(Path(img_path))]
        
        # Default label: [cat, dog, background]
        label = torch.zeros(3, dtype=torch.float)
        
        if source == 'coco':
            # COCO has specific category IDs
            ann_path = Path(img_path).with_name(f"{Path(img_path).stem.split('_')[0]}_ann.json")
            if ann_path.exists():
                with open(ann_path, 'r') as f:
                    anns = json.load(f)
                
                for ann in anns:
                    category_id = ann.get('category_id')
                    # In COCO, 17 is cat, 18 is dog
                    if category_id == 17:  # cat
                        label[0] = 1.0
                    elif category_id == 18:  # dog
                        label[1] = 1.0
        
        elif source == 'flickr':
            # Parse from title or tags
            title = metadata.get('title', '').lower()
            search_term = metadata.get('search_term', '').lower()
            
            if 'cat' in title or 'kitten' in title or 'cat' in search_term:
                label[0] = 1.0
            elif 'dog' in title or 'puppy' in title or 'dog' in search_term:
                label[1] = 1.0
        
        elif source == 'petfinder':
            # Petfinder has animal type
            animal_type = metadata.get('type', '').lower()
            
            if animal_type == 'cat':
                label[0] = 1.0
            elif animal_type == 'dog':
                label[1] = 1.0
        
        elif source == 'openimages':
            # Parse from the label file
            label_path = Path(img_path).with_suffix('.txt')
            if label_path.exists():
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) > 0:
                        class_name = parts[0]
                        if class_name == 'Cat':
                            label[0] = 1.0
                        elif class_name == 'Dog':
                            label[1] = 1.0
        
        # If neither cat nor dog confidently detected, mark as background
        if label[0] == 0 and label[1] == 0:
            label[2] = 1.0
            
        return label


class PseudoLabeler:
    """Generates pseudo-labels for unlabeled data using trained model."""
    
    def __init__(self, model, device, confidence_threshold=0.8, num_classes=3):
        """
        Initialize pseudo-labeler.
        
        Args:
            model: Trained model
            device: Device to run inference on
            confidence_threshold: Threshold for accepting pseudo-labels
            num_classes: Number of classes in the dataset
        """
        self.model = model
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.num_classes = num_classes
        
        # Set model to evaluation mode
        self.model.eval()
    
    def generate_pseudo_labels(self, dataset, output_dir):
        """
        Generate pseudo-labels for a dataset.
        
        Args:
            dataset: Dataset with unlabeled images
            output_dir: Directory to save pseudo-labels
            
        Returns:
            Dictionary mapping image paths to pseudo-label data
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)
        pseudo_labels = {}
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Generating pseudo-labels"):
                images = batch['image'].to(self.device)
                paths = batch['path']
                
                # Forward pass
                outputs = self.model(images)
                
                # Get class logits and segmentation maps
                logits = outputs['logits']
                seg_maps = outputs['segmentation_maps']
                
                # Apply softmax to get probabilities
                probs = F.softmax(logits, dim=1)
                max_probs, predicted_classes = torch.max(probs, dim=1)
                
                # Process each image in the batch
                for i, (path, prob, pred_class, seg_map) in enumerate(zip(paths, max_probs, predicted_classes, seg_maps)):
                    if prob >= self.confidence_threshold:
                        # Save pseudo-label data
                        pseudo_data = {
                            'label': pred_class.cpu().item(),
                            'confidence': prob.cpu().item(),
                            'class_probs': probs[i].cpu().tolist()
                        }
                        
                        pseudo_label_path = output_dir / f"{Path(path).stem}_pseudo.json"
                        with open(pseudo_label_path, 'w') as f:
                            json.dump(pseudo_data, f)
                        
                        # Save segmentation mask
                        mask = seg_map.argmax(dim=0).cpu().numpy().astype(np.uint8)
                        mask_img = Image.fromarray(mask * 85)  # Scale to visible range
                        mask_path = output_dir / f"{Path(path).stem}_pseudo_mask.png"
                        mask_img.save(mask_path)
                        
                        pseudo_labels[path] = pseudo_data
        
        logger.info(f"Generated {len(pseudo_labels)} pseudo-labels with confidence >= {self.confidence_threshold}")
        return pseudo_labels


class ConsistencyLoss(nn.Module):
    """Loss function to enforce consistency between predictions on augmented views."""
    
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
    
    def forward(self, pred1, pred2):
        """
        Calculate consistency loss between two predictions.
        
        Args:
            pred1: First prediction (logits or segmentation maps)
            pred2: Second prediction (logits or segmentation maps)
            
        Returns:
            Consistency loss value
        """
        # For logits, use KL divergence between softmaxed predictions
        if pred1.dim() == 2:
            p1 = F.softmax(pred1, dim=1)
            p2 = F.softmax(pred2, dim=1)
            
            loss = F.kl_div(p1.log(), p2, reduction='batchmean') + \
                   F.kl_div(p2.log(), p1, reduction='batchmean')
        
        # For segmentation maps, use MSE
        else:
            loss = F.mse_loss(pred1, pred2)
        
        return self.weight * loss


class CurriculumSampler:
    """Implements curriculum learning strategy for dataset sampling."""
    
    def __init__(self, datasets, max_samples=None, initial_ratio=0.8):
        """
        Initialize curriculum sampler.
        
        Args:
            datasets: Dictionary of datasets by type ('original', 'weak', 'pseudo', 'unlabeled')
            max_samples: Maximum samples to use from each dataset type (None = use all)
            initial_ratio: Initial ratio of original to additional data
        """
        self.datasets = datasets
        self.max_samples = max_samples
        self.initial_ratio = initial_ratio
        self.current_ratio = initial_ratio
        self.epoch = 0
        
        # Calculate dataset sizes
        self.dataset_sizes = {name: len(dataset) for name, dataset in datasets.items()}
        
        logger.info(f"Curriculum sampler initialized with datasets: {self.dataset_sizes}")
    
    def get_sampled_indices(self):
        """
        Get indices for sampling based on current curriculum stage.
        
        Returns:
            Dictionary mapping dataset names to lists of sampled indices
        """
        # Calculate current sample sizes based on curriculum stage
        orig_ratio = max(0.5, self.initial_ratio - (self.epoch * 0.05))
        self.current_ratio = orig_ratio
        
        # Calculate number of samples from each dataset
        total_additional = sum(self.dataset_sizes[name] for name in ['weak', 'pseudo', 'unlabeled'] 
                              if name in self.dataset_sizes)
        
        if total_additional == 0:
            logger.warning("No additional data available for curriculum learning")
            return {name: list(range(size)) for name, size in self.dataset_sizes.items()}
        
        orig_size = self.dataset_sizes.get('original', 0)
        if orig_size == 0:
            logger.warning("No original data available for curriculum learning")
            return {name: list(range(size)) for name, size in self.dataset_sizes.items()}
        
        # Calculate sample sizes
        total_samples = int(orig_size / orig_ratio)
        additional_samples = total_samples - orig_size
        
        # Prioritize datasets: pseudo > weak > unlabeled
        priority_order = ['pseudo', 'weak', 'unlabeled']
        additional_indices = {}
        remaining_samples = additional_samples
        
        for dataset_name in priority_order:
            if dataset_name in self.dataset_sizes:
                dataset_size = self.dataset_sizes[dataset_name]
                if dataset_size == 0:
                    additional_indices[dataset_name] = []
                    continue
                
                # Calculate proportion for this dataset
                if remaining_samples > 0:
                    samples_from_this = min(dataset_size, remaining_samples)
                    if self.max_samples:
                        samples_from_this = min(samples_from_this, self.max_samples)
                    
                    remaining_samples -= samples_from_this
                    
                    # Randomly sample indices
                    indices = random.sample(range(dataset_size), samples_from_this)
                    additional_indices[dataset_name] = indices
                else:
                    additional_indices[dataset_name] = []
        
        # Original data indices (use all)
        original_indices = list(range(orig_size))
        
        # Combine all indices
        all_indices = {'original': original_indices}
        all_indices.update(additional_indices)
        
        # Log sampling information
        sampled_counts = {name: len(indices) for name, indices in all_indices.items()}
        logger.info(f"Epoch {self.epoch}: Sampled {sampled_counts} indices with ratio {orig_ratio:.2f}")
        
        return all_indices
    
    def update_epoch(self):
        """Advance to the next epoch."""
        self.epoch += 1


class IntegratedDataset(Dataset):
    """Integrates original and additional datasets with curriculum learning."""
    
    def __init__(self, original_dataset, additional_datasets, transform=None):
        """
        Initialize integrated dataset.
        
        Args:
            original_dataset: Main dataset with strong supervision
            additional_datasets: Dictionary of additional datasets by type
            transform: Additional transforms to apply
        """
        self.original_dataset = original_dataset
        self.additional_datasets = additional_datasets
        self.transform = transform
        
        # Combine all datasets
        self.datasets = {'original': original_dataset}
        self.datasets.update(additional_datasets)
        
        # Initialize curriculum sampler
        self.sampler = CurriculumSampler(self.datasets)
        
        # Get initial indices
        self.sampled_indices = self.sampler.get_sampled_indices()
        
        # Calculate total size
        self.total_size = sum(len(indices) for indices in self.sampled_indices.values())
        
        logger.info(f"Integrated dataset created with {self.total_size} total samples")
    
    def __len__(self):
        return self.total_size
    
    def __getitem__(self, idx):
        """
        Get item from the integrated dataset.
        
        Args:
            idx: Index in the combined dataset
            
        Returns:
            Sample from one of the component datasets
        """
        # Map the index to a dataset and its internal index
        dataset_name, internal_idx = self._map_index(idx)
        
        # Get the dataset
        dataset = self.datasets[dataset_name]
        
        # Get the actual index in that dataset
        true_idx = self.sampled_indices[dataset_name][internal_idx]
        
        # Get the sample
        sample = dataset[true_idx]
        
        # Add source information
        if isinstance(sample, dict):
            sample['dataset_source'] = dataset_name
        else:
            # If not a dict, convert to dict
            if isinstance(sample, tuple):
                if len(sample) == 2:  # (image, label)
                    sample = {'image': sample[0], 'label': sample[1]}
                elif len(sample) == 3:  # (image, label, mask)
                    sample = {'image': sample[0], 'label': sample[1], 'mask': sample[2]}
            
            sample['dataset_source'] = dataset_name
        
        # Apply any additional transforms
        if self.transform and 'image' in sample:
            sample['image'] = self.transform(sample['image'])
        
        return sample
    
    def _map_index(self, idx):
        """
        Map a global index to a dataset name and local index.
        
        Args:
            idx: Global index
            
        Returns:
            Tuple of (dataset_name, local_index)
        """
        offset = 0
        for dataset_name, indices in self.sampled_indices.items():
            if idx < offset + len(indices):
                local_idx = idx - offset
                return dataset_name, local_idx
            offset += len(indices)
        
        # If we get here, the index is out of range
        raise IndexError(f"Index {idx} out of range for dataset of size {self.total_size}")
    
    def update_curriculum(self):
        """Update curriculum sampling for the next epoch."""
        self.sampler.update_epoch()
        self.sampled_indices = self.sampler.get_sampled_indices()
        self.total_size = sum(len(indices) for indices in self.sampled_indices.values())


def create_integrated_dataset(original_dataset, additional_data_dir, transform=None, 
                             model=None, device=None, generate_pseudo=True):
    """
    Create an integrated dataset with original and additional data.
    
    Args:
        original_dataset: Original dataset with strong supervision
        additional_data_dir: Directory with additional data
        transform: Transforms to apply to images
        model: Trained model for generating pseudo-labels
        device: Device to run model on
        generate_pseudo: Whether to generate pseudo-labels
        
    Returns:
        IntegratedDataset instance
    """
    additional_data_dir = Path(additional_data_dir)
    
    # Create datasets for weakly-labeled and unlabeled data
    weak_dataset = AdditionalDataset(
        additional_data_dir, 
        transform=transform, 
        label_type='weak'
    )
    
    unlabeled_dataset = AdditionalDataset(
        additional_data_dir, 
        transform=transform, 
        label_type='none'
    )
    
    # Generate pseudo-labels if requested
    pseudo_dataset = None
    if generate_pseudo and model is not None and device is not None:
        # Generate pseudo-labels
        pseudo_label_dir = additional_data_dir / 'pseudo_labels'
        
        # Create pseudo-labeler
        labeler = PseudoLabeler(model, device)
        
        # Generate pseudo-labels for unlabeled data
        pseudo_labels = labeler.generate_pseudo_labels(unlabeled_dataset, pseudo_label_dir)
        
        # Create dataset with pseudo-labels
        if pseudo_labels:
            pseudo_dataset = AdditionalDataset(
                additional_data_dir, 
                transform=transform, 
                label_type='pseudo'
            )
    
    # Collect all additional datasets
    additional_datasets = {
        'weak': weak_dataset,
        'unlabeled': unlabeled_dataset
    }
    
    if pseudo_dataset is not None:
        additional_datasets['pseudo'] = pseudo_dataset
    
    # Create integrated dataset
    return IntegratedDataset(original_dataset, additional_datasets, transform)


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate pseudo-labels for additional data')
    parser.add_argument('--data_dir', type=str, default='data/additional',
                        help='Directory with additional data')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--output_dir', type=str, default='data/additional/pseudo_labels',
                        help='Directory to save pseudo-labels')
    parser.add_argument('--confidence', type=float, default=0.8,
                        help='Confidence threshold for pseudo-labels')
    
    args = parser.parse_args()
    
    # Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(args.model_path, map_location=device)
    model.eval()
    
    # Create dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = AdditionalDataset(args.data_dir, transform=transform, label_type='none')
    
    # Generate pseudo-labels
    labeler = PseudoLabeler(model, device, confidence_threshold=args.confidence)
    pseudo_labels = labeler.generate_pseudo_labels(dataset, args.output_dir)
    
    print(f"Generated {len(pseudo_labels)} pseudo-labels with confidence >= {args.confidence}")
