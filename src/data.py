import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from pathlib import Path
import numpy as np
import urllib.request
import tarfile
import shutil
import random
import ssl
import logging

# Disable SSL verification for dataset download
ssl._create_default_https_context = ssl._create_unverified_context

class PetDataset(Dataset):
    def __init__(self, root_dir, split='train', weak_supervision=True, transform=None, test_split=0.2):
        self.root_dir = Path(root_dir)
        self.split = split
        self.weak_supervision = weak_supervision
        self.test_split = test_split
        
        logging.info(f"Initializing {split} dataset with weak_supervision={weak_supervision}")
        
        # Set transform pipeline
        if transform is None:
            self.transform = T.Compose([
                T.Resize((64, 64)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
            
        self.images_dir = self.root_dir / 'images'
        self.annotations_dir = self.root_dir / 'annotations'
        
        # Create split files and classes.txt if missing
        self._create_split_files()
        
        # Read image list for this split
        split_file = self.root_dir / f'{split}.txt'
        if not split_file.exists():
            raise ValueError(f"Invalid split name '{split}'. Expected one of ['train', 'val', 'test']")
        with open(split_file, 'r') as f:
            self.image_names = [line.strip() for line in f]
        
        # Read class list
        with open(self.root_dir / 'classes.txt', 'r') as f:
            self.classes = [line.strip() for line in f]
        
        logging.info(f"Dataset initialized with {len(self.image_names)} images and {len(self.classes)} classes")
            
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = self.images_dir / f'{img_name}.jpg'
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        if not self.weak_supervision:
            # Load full segmentation mask
            mask_path = self.annotations_dir / f'{img_name}.png'
            if not mask_path.exists():
                mask_path = self.annotations_dir / 'trimaps' / f'{img_name}.png'
            mask = Image.open(mask_path)
            mask = T.Resize((224, 224), interpolation=T.InterpolationMode.NEAREST)(mask)
            mask_np = np.array(mask, dtype=np.int64)
            # Oxford pet trimap: 1=pet, 2=border, 3=background
            pet_mask = np.zeros_like(mask_np)
            pet_mask[(mask_np == 1) | (mask_np == 2)] = 1
            mask = torch.from_numpy(pet_mask)
        else:
            # Extract breed name by dropping last '_<id>' suffix
            breed_name = '_'.join(img_name.split('_')[:-1])
            class_idx = self.classes.index(breed_name)
            mask = torch.tensor(class_idx)
        
        return {'image': image, 'mask': mask, 'image_name': img_name}
    
    def _create_split_files(self, test_split=0.2):
        """Create train/val/test split files and classes.txt if missing"""
        train_file = self.root_dir / 'train.txt'
        val_file = self.root_dir / 'val.txt'
        test_file = self.root_dir / 'test.txt'
        classes_file = self.root_dir / 'classes.txt'
        
        # Check existence
        if (train_file.exists() and train_file.stat().st_size > 0 and
            val_file.exists() and val_file.stat().st_size > 0 and
            test_file.exists() and test_file.stat().st_size > 0 and
            classes_file.exists() and classes_file.stat().st_size > 0):
            logging.info("Split files already exist and are not empty, skipping creation")
            return
        
        logging.info("Creating train/val/test split files")
        image_files = list(self.images_dir.glob('*.jpg'))
        image_names = [f.stem for f in image_files]
        
        # Write classes.txt with full breed names
        if not classes_file.exists() or classes_file.stat().st_size == 0:
            class_names = sorted(
                set('_'.join(name.split('_')[:-1]) for name in image_names)
            )
            with open(classes_file, 'w') as f:
                for cname in class_names:
                    f.write(f"{cname}\n")
            logging.info(f"Created classes.txt with {len(class_names)} classes")
        
        # Shuffle and split
        random.shuffle(image_names)
        n_total = len(image_names)
        n_test = int(test_split * n_total)
        n_val = int(test_split * n_total)
        n_train = n_total - n_val - n_test

        train_names = image_names[:n_train]
        val_names   = image_names[n_train:n_train + n_val]
        test_names  = image_names[n_train + n_val:]

        with open(train_file, 'w') as f:
            f.writelines(f"{n}\n" for n in train_names)
        with open(val_file, 'w') as f:
            f.writelines(f"{n}\n" for n in val_names)
        with open(test_file, 'w') as f:
            f.writelines(f"{n}\n" for n in test_names)

        logging.info(f"Created split files: {len(train_names)} train, {len(val_names)} val, {len(test_names)} test")

    @staticmethod
    def download_dataset(root_dir):
        """Download and unzip the Oxford-IIIT Pet dataset"""
        root_dir = Path(root_dir)
        root_dir.mkdir(parents=True, exist_ok=True)
        images_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
        ann_url    = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"
        logging.info("Downloading images...")
        img_tar = root_dir / 'images.tar.gz'
        urllib.request.urlretrieve(images_url, img_tar)
        logging.info("Extracting images...")
        with tarfile.open(img_tar, 'r:gz') as tar:
            tar.extractall(path=root_dir)
        logging.info("Downloading annotations...")
        ann_tar = root_dir / 'annotations.tar.gz'
        urllib.request.urlretrieve(ann_url, ann_tar)
        logging.info("Extracting annotations...")
        with tarfile.open(ann_tar, 'r:gz') as tar:
            tar.extractall(path=root_dir)
        img_tar.unlink(); ann_tar.unlink()
        # Organize files
        logging.info("Organizing dataset files...")
        (root_dir / 'images').mkdir(exist_ok=True)
        (root_dir / 'annotations').mkdir(exist_ok=True)
        for p in (root_dir / 'images').glob('*.jpg'):
            shutil.move(str(p), str((root_dir/'images')/p.name))
        for p in (root_dir / 'annotations').glob('*.png'):
            shutil.move(str(p), str((root_dir/'annotations')/p.name))
        logging.info("Cleanup complete. Creating split files...")
        PetDataset._create_split_files_static(root_dir, test_split=0.2)

    @staticmethod
    def _create_split_files_static(root_dir, test_split=0.2):
        """Static version of split creation, for download pipeline"""
        root_dir = Path(root_dir)
        images_dir = root_dir / 'images'
        train_file = root_dir / 'train.txt'
        val_file   = root_dir / 'val.txt'
        test_file  = root_dir / 'test.txt'
        classes_file = root_dir / 'classes.txt'

        if (train_file.exists() and train_file.stat().st_size > 0 and
            val_file.exists() and val_file.stat().st_size > 0 and
            test_file.exists() and test_file.stat().st_size > 0 and
            classes_file.exists() and classes_file.stat().st_size > 0):
            logging.info("Split files already exist, skipping static creation")
            return

        image_files = list(images_dir.glob('*.jpg'))
        image_names = [f.stem for f in image_files]
        
        if not classes_file.exists() or classes_file.stat().st_size == 0:
            class_names = sorted(
                set('_'.join(name.split('_')[:-1]) for name in image_names)
            )
            with open(classes_file, 'w') as f:
                for cname in class_names:
                    f.write(f"{cname}\n")
            logging.info(f"Created classes.txt with {len(class_names)} classes")

        random.shuffle(image_names)
        n_total = len(image_names)
        n_test  = int(test_split * n_total)
        n_val   = int(test_split * n_total)
        n_train = n_total - n_val - n_test

        train_names = image_names[:n_train]
        val_names   = image_names[n_train:n_train+n_val]
        test_names  = image_names[n_train+n_val:]

        with open(train_file, 'w') as f:
            f.writelines(f"{n}\n" for n in train_names)
        with open(val_file, 'w') as f:
            f.writelines(f"{n}\n" for n in val_names)
        with open(test_file, 'w') as f:
            f.writelines(f"{n}\n" for n in test_names)

        logging.info(f"Static split: {len(train_names)} train, {len(val_names)} val, {len(test_names)} test")