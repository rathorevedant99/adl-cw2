import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
import torchvision.transforms as T
from pathlib import Path
import numpy as np
import urllib.request
import tarfile
import shutil
import csv
import random
import ssl
import logging
import math
from torchvision.ops import masks_to_boxes
import json

# We might lose marks for this, but it's a pain to deal with the SSL certificate
ssl._create_default_https_context = ssl._create_unverified_context

class PetDataset(Dataset):
    def __init__(self, root_dir, split='train', weak_supervision=True, weak_supervision_types=None, 
                transform=None, test_split=0.2, scribble_density=0.05, subset_fraction=1.0):
        self.root_dir = Path(root_dir)
        self.split = split
        self.weak_supervision = weak_supervision
        self.weak_supervision_types = weak_supervision_types if weak_supervision_types else ['labels']
        self.test_split = test_split
        self.scribble_density = scribble_density
        self.subset_fraction = max(0.0, min(1.0, subset_fraction))  # Ensure it's between 0 and 1

        logging.info(f"Initializing {split} dataset with weak_supervision={weak_supervision}, types={weak_supervision_types}, subset_fraction={subset_fraction}")

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

        self.images_dir = self.root_dir / 'images'
        self.annotations_dir = self.root_dir / 'annotations'
        
        # Directories for generated weak supervision
        self.bboxes_dir = self.root_dir / 'bboxes'
        self.scribbles_dir = self.root_dir / 'scribbles'
        
        # Create directories if they don't exist
        self.bboxes_dir.mkdir(exist_ok=True, parents=True)
        self.scribbles_dir.mkdir(exist_ok=True, parents=True)

        # Ensure valid split
        split_file = self.root_dir / f'{split}.txt'
        if not split_file.exists():
            raise ValueError(f"Invalid split name '{split}'. Expected one of ['train', 'val', 'test']")

        with open(split_file, 'r') as f:
            self.image_names = [line.strip() for line in f.readlines()]
        
        # If subset_fraction < 1.0, sample a random subset
        if self.subset_fraction < 1.0 and len(self.image_names) > 0:
            import random
            num_samples = max(1, int(len(self.image_names) * self.subset_fraction))
            self.image_names = random.sample(self.image_names, num_samples)
            logging.info(f"Using a subset of {num_samples} images ({self.subset_fraction:.1%} of the original dataset)")
        
        with open(self.root_dir / 'classes.txt', 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

        logging.info(f"Dataset initialized with {len(self.image_names)} images and {len(self.classes)} classes")
        
        # Only generate weak supervision data if needed
        if 'bboxes' in self.weak_supervision_types:
            self._ensure_bboxes_generated()
            
        if 'scribbles' in self.weak_supervision_types:
            self._ensure_scribbles_generated()

        # Ensure valid split
        split_file = self.root_dir / f'{split}.txt'
        if not split_file.exists():
            raise ValueError(f"Invalid split name '{split}'. Expected one of ['train', 'val', 'test']")

        with open(split_file, 'r') as f:
            self.image_names = [line.strip() for line in f.readlines()]

        with open(self.root_dir / 'classes.txt', 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

        logging.info(f"Dataset initialized with {len(self.image_names)} images and {len(self.classes)} classes")

            
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        
        img_path = self.images_dir / f'{img_name}.jpg'
        image = Image.open(img_path).convert('RGB')
        
        # Always prepare the full segmentation mask for evaluation purposes
        #mask_path = self.annotations_dir / f'{img_name}.png'
        mask_path = self.annotations_dir / 'trimaps' / f'{img_name}.png'
        full_mask = Image.open(mask_path)
        full_mask = T.Resize((224, 224))(full_mask)
        full_mask_tensor = torch.from_numpy(np.array(full_mask)).long()
        
        # Apply transforms to image
        image_tensor = self.transform(image)
        
        result = {
            'image': image_tensor,
            'image_name': img_name,
            'full_mask': full_mask_tensor  # Always include full mask for evaluation
        }
        
        # For weak supervision, prepare appropriate supervision signals
        if self.weak_supervision:
            # Image-level labels (always include for weak supervision)
            class_name = img_name.split('_')[0]
            class_idx = self.classes.index(class_name)
            result['class_label'] = torch.tensor(class_idx)
            
            # Use specific weak supervision types
            if 'labels' in self.weak_supervision_types:
                result['mask'] = torch.tensor(class_idx)  # Just the class label
                
            if 'bboxes' in self.weak_supervision_types:
                bbox_path = self.bboxes_dir / f'{img_name}.json'
                with open(bbox_path, 'r') as f:
                    bbox = json.load(f)  # [x1, y1, x2, y2]
                
                # Convert to normalized bbox [x1/W, y1/H, x2/W, y2/H]
                h, w = image.height, image.width
                normalized_bbox = [bbox[0]/w, bbox[1]/h, bbox[2]/w, bbox[3]/h]
                result['bbox'] = torch.tensor(normalized_bbox)
                
                # Create bbox mask (1 inside bbox, 0 outside)
                bbox_mask = torch.zeros((224, 224))
                x1, y1, x2, y2 = int(normalized_bbox[0]*224), int(normalized_bbox[1]*224), \
                                int(normalized_bbox[2]*224), int(normalized_bbox[3]*224)
                bbox_mask[y1:y2, x1:x2] = 1
                result['bbox_mask'] = bbox_mask
                
            if 'scribbles' in self.weak_supervision_types:
                scribble_path = self.scribbles_dir / f'{img_name}.pt'
                scribble_mask = torch.load(scribble_path)
                result['scribble_mask'] = scribble_mask
            
            # Combine all weak supervisions into a single mask
            # If no weak supervision is specified, just use the class label
            if not any(k in result for k in ['mask', 'bbox_mask', 'scribble_mask']):
                result['mask'] = torch.tensor(class_idx)
            else:
                # Priority order: scribbles > bbox > label
                if 'scribble_mask' in result:
                    result['mask'] = result['scribble_mask']
                elif 'bbox_mask' in result:
                    result['mask'] = result['bbox_mask']
                elif 'labels' in self.weak_supervision_types:
                    result['mask'] = torch.tensor(class_idx)
        else:
            # For full supervision, use the full segmentation mask
            result['mask'] = full_mask_tensor
        
        return result
    
    def _ensure_bboxes_generated(self):
        """Generate bounding boxes for the dataset if not already present"""
        # Check if we already have all bboxes
        if all((self.bboxes_dir / f'{img_name}.json').exists() for img_name in self.image_names):
            logging.info("All bounding boxes already generated")
            return
        
        logging.info("Generating bounding boxes...")
        for img_name in self.image_names:
            bbox_path = self.bboxes_dir / f'{img_name}.json'
            if bbox_path.exists():
                continue
                
            # Load mask to generate bbox
            #mask_path = self.annotations_dir / f'{img_name}.png'
            mask_path = self.annotations_dir / 'trimaps' / f'{img_name}.png'
            mask = Image.open(mask_path)
            mask_tensor = torch.from_numpy(np.array(mask)).unsqueeze(0)
            
            # Get foreground pixels (where mask > 0)
            foreground = (mask_tensor > 0).float()
            
            # Handle empty masks
            if foreground.sum() == 0:
                # If no foreground, use the whole image
                h, w = mask.height, mask.width
                bbox = [0, 0, w, h]
            else:
                # Compute bounding box from mask
                bbox = masks_to_boxes(foreground)[0].tolist()
            
            # Save bbox
            with open(bbox_path, 'w') as f:
                json.dump(bbox, f)
        
        logging.info("Bounding box generation completed")
    
    def _ensure_scribbles_generated(self):
        """Generate scribbles for the dataset if not already present"""
        # Check if we already have all scribbles
        if all((self.scribbles_dir / f'{img_name}.pt').exists() for img_name in self.image_names):
            logging.info("All scribbles already generated")
            return
        
        logging.info("Generating scribbles...")
        for img_name in self.image_names:
            scribble_path = self.scribbles_dir / f'{img_name}.pt'
            if scribble_path.exists():
                continue
                
            # Load mask to generate scribbles
            #mask_path = self.annotations_dir / f'{img_name}.png'
            mask_path = self.annotations_dir / 'trimaps' / f'{img_name}.png'
            mask = Image.open(mask_path)
            mask_np = np.array(mask)
            
            # Get unique class values
            unique_classes = np.unique(mask_np)
            
            # Create scribble mask
            h, w = mask_np.shape
            scribble_mask = np.zeros((h, w), dtype=np.uint8)
            
            # For each class, generate random scribbles
            for cls in unique_classes:
                if cls == 0:  # Skip background
                    continue
                    
                # Find pixels belonging to this class
                class_pixels = np.where(mask_np == cls)
                if len(class_pixels[0]) == 0:
                    continue
                    
                # Select a random subset of pixels
                num_pixels = len(class_pixels[0])
                num_scribble_pixels = max(3, int(num_pixels * self.scribble_density))
                
                # Ensure we don't try to sample more pixels than exist
                num_scribble_pixels = min(num_scribble_pixels, num_pixels)
                
                # Randomly select scribble pixels
                indices = np.random.choice(num_pixels, num_scribble_pixels, replace=False)
                y_coords = class_pixels[0][indices]
                x_coords = class_pixels[1][indices]
                
                # Set scribble pixels
                scribble_mask[y_coords, x_coords] = cls
            
            # Resize scribble mask to 224x224
            scribble_mask_pil = Image.fromarray(scribble_mask)
            scribble_mask_resized = T.Resize((224, 224))(scribble_mask_pil)
            scribble_mask_tensor = torch.from_numpy(np.array(scribble_mask_resized)).long()
            
            # Save scribbles
            torch.save(scribble_mask_tensor, scribble_path)
        
        logging.info("Scribble generation completed")
    
    def visualize_weak_supervision(self, img_name, output_dir="weak_supervision_vis"):
        """Visualize different weak supervision signals for a specific image"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        img_path = self.images_dir / f'{img_name}.jpg'
        mask_path = self.annotations_dir / f'{img_name}.png'
        
        # Load image and mask
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)
        
        # Resize for consistency
        image = image.resize((224, 224))
        mask = mask.resize((224, 224))
        
        # Save original image and mask
        image.save(output_dir / f"{img_name}_original.jpg")
        mask.save(output_dir / f"{img_name}_mask.png")
        
        # Visualize bounding box
        if 'bboxes' in self.weak_supervision_types:
            bbox_path = self.bboxes_dir / f'{img_name}.json'
            with open(bbox_path, 'r') as f:
                bbox = json.load(f)  # [x1, y1, x2, y2]
            
            # Draw bbox on image
            bbox_vis = image.copy()
            draw = ImageDraw.Draw(bbox_vis)
            
            # Scale bbox to 224x224
            orig_w, orig_h = Image.open(img_path).size
            x1 = int(bbox[0] * 224 / orig_w)
            y1 = int(bbox[1] * 224 / orig_h)
            x2 = int(bbox[2] * 224 / orig_w)
            y2 = int(bbox[3] * 224 / orig_h)
            
            draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
            bbox_vis.save(output_dir / f"{img_name}_bbox.jpg")
        
        # Visualize scribbles
        if 'scribbles' in self.weak_supervision_types:
            scribble_path = self.scribbles_dir / f'{img_name}.pt'
            scribble_mask = torch.load(scribble_path).numpy()
            
            # Create RGB representation of scribbles
            scribble_vis = np.zeros((224, 224, 3), dtype=np.uint8)
            
            # Set different colors for different classes
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
            
            for cls in range(1, 6):  # Assuming up to 5 classes
                cls_pixels = scribble_mask == cls
                if cls_pixels.any():
                    color_idx = (cls - 1) % len(colors)
                    scribble_vis[cls_pixels] = colors[color_idx]
            
            scribble_img = Image.fromarray(scribble_vis)
            
            # Overlay scribbles on original image
            overlay = Image.blend(image.convert('RGB'), scribble_img, 0.7)
            overlay.save(output_dir / f"{img_name}_scribbles.jpg")
        
        logging.info(f"Visualizations saved to {output_dir}")
    
    @staticmethod
    def _create_split_files_static(root_dir, test_split=0.2):
        """Static method to create train/val/test splits during download"""
        root_dir = Path(root_dir)
        train_file = root_dir / 'train.txt'
        val_file = root_dir / 'val.txt'
        test_file = root_dir / 'test.txt'
        classes_file = root_dir / 'classes.txt'

        if all(f.exists() and f.stat().st_size > 0 for f in [train_file, val_file, test_file, classes_file]):
            logging.info("Split files already exist and are not empty, skipping creation")
            return

        logging.info("Creating train/val/test split files")
        images_dir = root_dir / 'images'
        image_files = list(images_dir.glob('*.jpg'))
        image_names = [f.stem for f in image_files]

        class_names = sorted(set(name.split('_')[0] for name in image_names))
        with open(classes_file, 'w') as f:
            for class_name in class_names:
                f.write(f"{class_name}\n")
        logging.info(f"Created classes.txt with {len(class_names)} classes")

        random.shuffle(image_names)
        n_total = len(image_names)
        n_test = int(test_split * n_total)
        n_val = int(test_split * n_total)
        n_train = n_total - n_val - n_test

        train_names = image_names[:n_train]
        val_names = image_names[n_train:n_train + n_val]
        test_names = image_names[n_train + n_val:]

        with open(train_file, 'w') as f:
            for name in train_names:
                f.write(f"{name}\n")
        with open(val_file, 'w') as f:
            for name in val_names:
                f.write(f"{name}\n")
        with open(test_file, 'w') as f:
            for name in test_names:
                f.write(f"{name}\n")

        logging.info(f"Created split files: {len(train_names)} train, {len(val_names)} val, {len(test_names)} test")
    
    @staticmethod
    def download_dataset(root_dir):
        """Download and extract the Oxford-IIIT Pet dataset"""
        root_dir = Path(root_dir)
        root_dir.mkdir(parents=True, exist_ok=True)
        
        # URLs for the dataset
        images_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
        annotations_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"
        
        # Download and extract images
        logging.info("Downloading images...")
        images_path = root_dir / "images.tar.gz"
        urllib.request.urlretrieve(images_url, images_path)
        
        logging.info("Extracting images...")
        with tarfile.open(images_path, 'r:gz') as tar:
            tar.extractall(path=root_dir)
        
        # Download and extract annotations
        logging.info("Downloading annotations...")
        annotations_path = root_dir / "annotations.tar.gz"
        urllib.request.urlretrieve(annotations_url, annotations_path)
        
        logging.info("Extracting annotations...")
        with tarfile.open(annotations_path, 'r:gz') as tar:
            tar.extractall(path=root_dir)
        
        # Clean up tar files
        images_path.unlink()
        annotations_path.unlink()
        
        # Move files to correct locations
        logging.info("Organizing dataset files...")
        images_dir = root_dir / "images"
        annotations_dir = root_dir / "annotations"
        
        # Create directories if they don't exist
        images_dir.mkdir(exist_ok=True)
        annotations_dir.mkdir(exist_ok=True)
        
        # Move image files
        for img_path in (root_dir / "images").glob("*.jpg"):
            shutil.move(str(img_path), str(images_dir / img_path.name))
            
        # Move annotation files
        for ann_path in (root_dir / "annotations").glob("*.png"):
            shutil.move(str(ann_path), str(annotations_dir / ann_path.name))
            
        logging.info("Dataset download and organization completed")

        # Create split files after downloading
        logging.info("Creating train/val/test split files after download")
        PetDataset._create_split_files_static(root_dir, test_split=0.2)