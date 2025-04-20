import torch
import torchvision.transforms.functional as F
import random
from torch.utils.data import Dataset
import logging
from pathlib import Path
import torchvision.utils as vutils
import matplotlib.pyplot as plt

class DataAugmentation:
    def __init__(self, p=0.5):
        """
        Initialize the data augmentation transform.
        
        Args:
            p (float): Probability of applying each augmentation
        """
        self.p = p
    
    def __call__(self, image, mask):
        """
        Apply random augmentations to both image and mask.
        
        Args:
            image (torch.Tensor): Image tensor of shape [C, H, W]
            mask (torch.Tensor): Mask tensor (either segmentation mask or class index)
            
        Returns:
            tuple: (augmented_image, augmented_mask)
        """
        # Random horizontal flip
        if random.random() < self.p:
            image = F.hflip(image)
            if mask.dim() > 0:  # Only flip mask if it's a segmentation mask
                mask = F.hflip(mask)
        
        # Random vertical flip
        if random.random() < self.p:
            image = F.vflip(image)
            if mask.dim() > 0:
                mask = F.vflip(mask)
        
        # Random rotation
        if random.random() < self.p:
            angle = random.choice([90, 180, 270])
            image = F.rotate(image, angle)
            if mask.dim() > 0:
                mask = F.rotate(mask, angle)
        
        # Color augmentations (only for image)
        if random.random() < self.p:
            image = F.adjust_brightness(image, random.uniform(0.8, 1.2))
            image = F.adjust_contrast(image, random.uniform(0.8, 1.2))
            image = F.adjust_saturation(image, random.uniform(0.8, 1.2))
            image = F.adjust_hue(image, random.uniform(-0.1, 0.1))
        
        return image, mask

class AugmentedDataset(Dataset):
    def __init__(self, original_dataset):
        """
        Create an augmented version of the original dataset.
        Only returns augmented images when augmentation was actually applied.
        
        Args:
            original_dataset: The dataset to augment
        """
        self.dataset = original_dataset
        self.augmentation = DataAugmentation(p=0.5)
        self.augmented_indices = []  # Store indices of images that were actually augmented
    
    def __len__(self):
        return len(self.augmented_indices)
    
    def __getitem__(self, idx):
        dataset_idx = self.augmented_indices[idx]
        sample = self.dataset[dataset_idx]
        aug_image, aug_mask = self.augmentation(sample['image'], sample['mask'])
        
        return {
            'image': aug_image,
            'mask': aug_mask,
            'image_name': sample['image_name']
        }
    
    def _build_augmented_indices(self):
        """Build list of indices that were actually augmented"""
        self.augmented_indices = []
        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]
            aug_image, aug_mask = self.augmentation(sample['image'], sample['mask'])
            
            if not torch.equal(sample['image'], aug_image):
                self.augmented_indices.append(idx)
        
        logging.info(f"Created augmented dataset with {len(self.augmented_indices)} augmented images")
    
    def save_sample_pairs(self, num_samples=5, save_dir='augmentation_samples'):
        """Save sample pairs of original and augmented images"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if len(self.augmented_indices) < num_samples:
            num_samples = len(self.augmented_indices)
            logging.warning(f"Only {num_samples} augmented samples available")
        
        sample_indices = random.sample(self.augmented_indices, num_samples)
        
        for idx, dataset_idx in enumerate(sample_indices):
            original_sample = self.dataset[dataset_idx]
            aug_image, aug_mask = self.augmentation(original_sample['image'], original_sample['mask'])
            
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            
            original_img = original_sample['image'] * std + mean
            augmented_img = aug_image * std + mean
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            
            # Plot original image
            ax1.imshow(original_img.permute(1, 2, 0).cpu().numpy())
            ax1.set_title('Original Image')
            ax1.axis('off')
            
            # Plot augmented image
            ax2.imshow(augmented_img.permute(1, 2, 0).cpu().numpy())
            ax2.set_title('Augmented Image')
            ax2.axis('off')
            
            # Save the figure
            plt.savefig(save_dir / f'sample_pair_{idx}.png')
            plt.close()
            
            logging.info(f"Saved sample pair {idx} with image name: {original_sample['image_name']}")
        
        logging.info(f"Saved {num_samples} sample pairs to {save_dir}")
    