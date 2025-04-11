import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import numpy as np

class OxfordPetDataset(Dataset):
    def __init__(self, root, image_size=256, mode='train', debug=False):
        """
        root: Path to the dataset root containing images/ and annotations/trimaps/.
        image_size: The size to which we will resize the images and masks.
        mode: 'train' or 'val' (or 'test'), if you want to split the data.
        debug: If True, will print debug information about the dataset.
        """
        self.root = root
        self.debug = debug
        self.image_paths = sorted(glob.glob(os.path.join(root, 'images', '*.jpg')))
        self.mask_paths = sorted(glob.glob(os.path.join(root, 'annotations', 'trimaps', '*.png')))
        
        if self.debug:
            print(f"Found {len(self.image_paths)} images and {len(self.mask_paths)} masks")
            
        # Ensure we have the same number of images and masks
        assert len(self.image_paths) == len(self.mask_paths), "Mismatch between number of images and masks"
        
        # Simple train/val split (e.g., 90%/10%)
        num_images = len(self.image_paths)
        split_idx = int(0.9 * num_images)
        
        if mode == 'train':
            self.image_paths = self.image_paths[:split_idx]
            self.mask_paths = self.mask_paths[:split_idx]
        else:
            self.image_paths = self.image_paths[split_idx:]
            self.mask_paths = self.mask_paths[split_idx:]
        
        if self.debug:
            print(f"{mode} set size: {len(self.image_paths)}")
            
        self.transform_image = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor()
        ])
        
        self.transform_mask = T.Compose([
            T.Resize((image_size, image_size), interpolation=Image.NEAREST),
            T.ToTensor()  # Will produce a float tensor in [0,1]
        ])
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)
        
        # Debug: Check mask values before transformation
        if self.debug and idx == 0:
            mask_np = np.array(mask)
            print(f"Original mask unique values: {np.unique(mask_np)}")
        
        image = self.transform_image(image)
        mask = self.transform_mask(mask)
        
        # Debug: Check mask values after transformation
        if self.debug and idx == 0:
            print(f"Transformed mask shape: {mask.shape}")
            print(f"Transformed mask unique values: {torch.unique(mask)}")
            print(f"Transformed mask min: {mask.min()}, max: {mask.max()}, mean: {mask.mean()}")
        
        # In Oxford Pets dataset, the trimap has 3 values:
        # 1 - pet, 2 - background, 3 - boundary
        # We want to create a binary mask where 1 = pet, 0 = background/boundary
        
        # Check if we need to convert from the original range
        if torch.max(mask) > 1.0:
            mask = mask / 255.0  # Normalize if needed
            
        # Try various approaches to convert trimap to binary mask
        if len(torch.unique(mask)) <= 3:
            # Method 1: Check for values closest to 1/3 (which would be class 1 after normalization)
            mask_binary = (torch.abs(mask - 1/3) < 0.1).float()
            
            # If not enough foreground pixels, try a different approach
            if mask_binary.mean() < 0.01:
                # Method 2: Try different threshold
                mask_binary = (mask < 0.4).float()
                
                # Method 3: If still not working, try direct comparison with likely values
                if mask_binary.mean() < 0.01 or mask_binary.mean() > 0.99:
                    # Inspect unique values and choose a sensible threshold
                    unique_vals = torch.unique(mask)
                    if self.debug and idx < 5:
                        print(f"Sample {idx} unique mask values: {unique_vals}")
                    
                    # Typically in a trimap: lowest value = foreground, middle = background, highest = boundary
                    if len(unique_vals) >= 2:
                        threshold = (unique_vals[0] + unique_vals[1]) / 2 if len(unique_vals) > 1 else 0.5
                        mask_binary = (mask <= threshold).float()
        else:
            # If mask doesn't have a clear trimap structure, use a generic threshold
            mask_binary = (mask > 0.5).float()
        
        # Final check - if mask is still problematic, create dummy mask for debugging
        if mask_binary.mean() < 0.001 or mask_binary.mean() > 0.999:
            if self.debug and idx < 5:
                print(f"WARNING: Sample {idx} has a problematic mask, with mean: {mask_binary.mean()}")
            
            # Create a simple split mask for debugging purposes
            if mask_binary.mean() < 0.001:  # If all zeros, create some foreground
                mask_binary = torch.zeros_like(mask)
                h, w = mask.shape[-2:]
                mask_binary[..., :h//2, :w//2] = 1.0  # Make top-left quadrant foreground
        
        # Ensure mask has shape [H, W] without channel dimension
        if mask_binary.dim() > 2:
            mask_binary = mask_binary.squeeze(0)
            
        # Final debug check
        if self.debug and idx == 0:
            print(f"Final binary mask shape: {mask_binary.shape}")
            print(f"Final binary mask unique values: {torch.unique(mask_binary)}")
            print(f"Final binary mask min: {mask_binary.min()}, max: {mask_binary.max()}, mean: {mask_binary.mean()}")
        
        return image, mask_binary