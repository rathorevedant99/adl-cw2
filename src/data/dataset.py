import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from pathlib import Path
import numpy as np

class PetDataset(Dataset):
    def __init__(self, root_dir, split='train', weak_supervision=True, transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.weak_supervision = weak_supervision
        
        # Default transforms
        if transform is None:
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
            
        # Load image paths and labels
        self.images_dir = self.root_dir / 'images'
        self.annotations_dir = self.root_dir / 'annotations'
        
        # Load split file
        split_file = self.root_dir / f'{split}.txt'
        with open(split_file, 'r') as f:
            self.image_names = [line.strip() for line in f.readlines()]
            
        # Load class names
        with open(self.root_dir / 'classes.txt', 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
            
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        
        # Load image
        img_path = self.images_dir / f'{img_name}.jpg'
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        # Load annotation if not using weak supervision
        if not self.weak_supervision:
            mask_path = self.annotations_dir / f'{img_name}.png'
            mask = Image.open(mask_path)
            mask = T.Resize((224, 224))(mask)
            mask = torch.from_numpy(np.array(mask)).long()
        else:
            # For weak supervision, we only use image-level labels
            # In this case, we'll use the class name from the image name
            class_name = img_name.split('_')[0]
            class_idx = self.classes.index(class_name)
            mask = torch.tensor(class_idx)
        
        return {
            'image': image,
            'mask': mask,
            'image_name': img_name
        }
    
    @staticmethod
    def download_dataset(root_dir):
        """Download and extract the Oxford-IIIT Pet dataset"""
        import urllib.request
        import tarfile
        
        root_dir = Path(root_dir)
        root_dir.mkdir(parents=True, exist_ok=True)
        
        # Download dataset
        url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
        tar_path = root_dir / "images.tar.gz"
        
        if not tar_path.exists():
            print("Downloading dataset...")
            urllib.request.urlretrieve(url, tar_path)
            
            # Extract dataset
            print("Extracting dataset...")
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(path=root_dir)
            
            # Clean up
            tar_path.unlink()
            
        print("Dataset downloaded and extracted successfully!") 