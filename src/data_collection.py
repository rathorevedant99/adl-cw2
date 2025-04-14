"""
Data collection utilities for acquiring additional weakly-labeled and unlabeled data.
"""

import os
import json
import requests
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import shutil
from io import BytesIO
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataCollector:
    """Collects additional weakly-labeled data for pet segmentation."""
    
    def __init__(self, 
                 output_dir, 
                 flickr_api_key=None, 
                 flickr_api_secret=None):
        """
        Initialize data collector.
        
        Args:
            output_dir: Directory to save collected data
            flickr_api_key: API key for Flickr (optional)
            flickr_api_secret: API secret for Flickr (optional)
        """
        self.output_dir = Path(output_dir)
        self.flickr_credentials = {
            'api_key': flickr_api_key,
            'api_secret': flickr_api_secret
        }
        
        # Create output directories
        self.flickr_dir = self.output_dir / 'flickr'
        self.openimages_dir = self.output_dir / 'openimages'
        self.coco_dir = self.output_dir / 'coco'
        
        for directory in [self.flickr_dir, self.openimages_dir, self.coco_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def collect_flickr_images(self, search_terms, max_images=500, per_page=100):
        """
        Collect images from Flickr with Creative Commons licenses.
        
        Args:
            search_terms: List of search terms to use
            max_images: Maximum number of images to download
            per_page: Number of images per API request
        """
        if not all(self.flickr_credentials.values()):
            logger.warning("Flickr API credentials not provided. Skipping Flickr collection.")
            return []
            
        try:
            import flickrapi
        except ImportError:
            logger.error("flickrapi not installed. Please install with 'pip install flickrapi'")
            return []
        
        api_key = self.flickr_credentials['api_key']
        api_secret = self.flickr_credentials['api_secret']
        
        flickr = flickrapi.FlickrAPI(api_key, api_secret, format='parsed-json')
        downloaded_images = []
        
        for term in search_terms:
            logger.info(f"Searching Flickr for: {term}")
            
            # Search for images with CC licenses
            for page in range(1, (max_images // per_page) + 2):
                try:
                    photos = flickr.photos.search(
                        text=term,
                        per_page=per_page,
                        page=page,
                        license='1,2,4,5,7',  # Creative Commons licenses
                        content_type=1,  # Photos only
                        media='photos',
                        sort='relevance',
                        extras='url_o,license,owner_name'
                    )
                    
                    if not photos or 'photos' not in photos or 'photo' not in photos['photos']:
                        logger.warning(f"No photos found for {term} on page {page}")
                        break
                    
                    for photo in tqdm(photos['photos']['photo'], desc=f"Downloading {term} (page {page})"):
                        if 'url_o' in photo:
                            url = photo['url_o']
                        else:
                            url = f"https://farm{photo['farm']}.staticflickr.com/{photo['server']}/{photo['id']}_{photo['secret']}.jpg"
                        
                        try:
                            img_path = self.flickr_dir / f"{photo['id']}.jpg"
                            if not img_path.exists():
                                response = requests.get(url, timeout=10)
                                if response.status_code == 200:
                                    img = Image.open(BytesIO(response.content))
                                    img.save(img_path)
                                    
                                    # Save metadata
                                    metadata = {
                                        'id': photo['id'],
                                        'title': photo.get('title', ''),
                                        'owner': photo.get('owner_name', ''),
                                        'license': photo.get('license', ''),
                                        'search_term': term,
                                        'url': url
                                    }
                                    
                                    meta_path = self.flickr_dir / f"{photo['id']}_meta.json"
                                    with open(meta_path, 'w') as f:
                                        json.dump(metadata, f)
                                    
                                    downloaded_images.append(str(img_path))
                                    
                                    # Be nice to the API
                                    time.sleep(0.5)
                        except Exception as e:
                            logger.error(f"Error downloading {url}: {e}")
                    
                    # Check if we've reached the maximum
                    if len(downloaded_images) >= max_images:
                        break
                        
                except Exception as e:
                    logger.error(f"Error fetching page {page} for {term}: {e}")
                    break
                
            # Check if we've reached the maximum across all search terms
            if len(downloaded_images) >= max_images:
                logger.info(f"Reached maximum number of images ({max_images})")
                break
        
        logger.info(f"Downloaded {len(downloaded_images)} images from Flickr")
        return downloaded_images
    

    
    def download_coco_pets(self, coco_path, max_images=500):
        """
        Extract pet-related images from COCO dataset.
        
        Args:
            coco_path: Path to COCO dataset
            max_images: Maximum number of images to copy
        """
        try:
            from pycocotools.coco import COCO
        except ImportError:
            logger.error("pycocotools not installed. Please install with 'pip install pycocotools'")
            return []
        
        coco_path = Path(coco_path)
        ann_file = coco_path / "annotations" / "instances_train2017.json"
        
        if not ann_file.exists():
            logger.error(f"COCO annotation file not found at {ann_file}")
            return []
        
        try:
            # Initialize COCO API
            coco = COCO(ann_file)
            
            # Pet categories in COCO
            cat_ids = coco.getCatIds(catNms=['dog', 'cat'])
            img_ids = []
            
            for cat_id in cat_ids:
                # Get image IDs containing this category
                cat_img_ids = coco.getImgIds(catIds=[cat_id])
                img_ids.extend(cat_img_ids)
            
            # Remove duplicates
            img_ids = list(set(img_ids))
            
            # Limit to max_images
            if len(img_ids) > max_images:
                img_ids = img_ids[:max_images]
            
            logger.info(f"Found {len(img_ids)} pet images in COCO dataset")
            
            # Download/copy images and associated annotations
            downloaded_images = []
            for img_id in tqdm(img_ids, desc="Processing COCO images"):
                img_info = coco.loadImgs(img_id)[0]
                src_path = coco_path / "train2017" / img_info['file_name']
                
                if not src_path.exists():
                    logger.warning(f"Source image not found: {src_path}")
                    continue
                
                dst_path = self.coco_dir / img_info['file_name']
                
                # Copy the image
                shutil.copy(src_path, dst_path)
                
                # Get annotations for this image (including bounding boxes)
                ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids, iscrowd=False)
                anns = coco.loadAnns(ann_ids)
                
                # Save annotations
                ann_path = self.coco_dir / f"{img_id}_ann.json"
                with open(ann_path, 'w') as f:
                    json.dump(anns, f)
                
                downloaded_images.append(str(dst_path))
            
            logger.info(f"Copied {len(downloaded_images)} pet images from COCO dataset")
            return downloaded_images
            
        except Exception as e:
            logger.error(f"Error processing COCO dataset: {e}")
            return []
    
    def download_openimages_pets(self, oid_dir, max_images=500):
        """
        Download pet images from Open Images Dataset.
        
        Args:
            oid_dir: Directory to save OID downloads
            max_images: Maximum number of images to download
        """
        try:
            oid_dir = Path(oid_dir)
            oid_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if OIDv4 toolkit is installed
            try:
                import sys
                import subprocess
                
                # Clone OIDv4 toolkit if not already present
                oidv4_dir = oid_dir / "OIDv4_ToolKit"
                if not oidv4_dir.exists():
                    logger.info("Cloning OIDv4 toolkit...")
                    subprocess.run(
                        ["git", "clone", "https://github.com/EscVM/OIDv4_ToolKit.git"],
                        cwd=oid_dir
                    )
                
                # Install requirements
                requirements_file = oidv4_dir / "requirements.txt"
                if requirements_file.exists():
                    logger.info("Installing OIDv4 requirements...")
                    subprocess.run(
                        ["pip", "install", "-r", "requirements.txt"],
                        cwd=oidv4_dir
                    )
                
                # Download pet-related classes
                pet_classes = ["Dog", "Cat", "Pet"]
                for pet_class in pet_classes:
                    logger.info(f"Downloading {pet_class} images from Open Images...")
                    
                    # Calculate limit per class
                    limit_per_class = max_images // len(pet_classes)
                    
                    subprocess.run([
                        "python", "main.py", "downloader",
                        "--classes", pet_class,
                        "--type", "all",
                        "--limit", str(limit_per_class),
                        "--yes"
                    ], cwd=oidv4_dir)
                
                # Copy the downloaded images to our directory
                downloaded_images = []
                source_dirs = [
                    oidv4_dir / "OID" / "Dataset" / "train" / pet_class
                    for pet_class in pet_classes
                ]
                
                for source_dir in source_dirs:
                    if not source_dir.exists():
                        continue
                        
                    for img_path in source_dir.glob("*.jpg"):
                        dest_path = self.openimages_dir / img_path.name
                        shutil.copy(img_path, dest_path)
                        
                        # Also copy label file if exists
                        label_path = img_path.with_suffix('.txt')
                        if label_path.exists():
                            shutil.copy(label_path, dest_path.with_suffix('.txt'))
                        
                        downloaded_images.append(str(dest_path))
                
                logger.info(f"Copied {len(downloaded_images)} pet images from Open Images")
                return downloaded_images
                
            except Exception as e:
                logger.error(f"Error with OIDv4 toolkit: {e}")
                return []
                
        except Exception as e:
            logger.error(f"Error downloading from Open Images: {e}")
            return []
    
    def collect_all_sources(self, max_per_source=500):
        """
        Collect data from all available sources.
        
        Args:
            max_per_source: Maximum images to collect from each source
            
        Returns:
            Dictionary with collected image paths by source
        """
        all_collected = {}
        
        # Flickr (weakly labeled - tags/titles as weak labels)
        search_terms = [
            "cat", "dog", "kitten", "puppy", "pet cat", "pet dog",
            "cat portrait", "dog portrait", "cat face", "dog face"
        ]
        flickr_images = self.collect_flickr_images(search_terms, max_images=max_per_source)
        all_collected['flickr'] = flickr_images
        
        # Check for COCO dataset path (user would need to provide)
        coco_path = os.environ.get('COCO_PATH')
        if coco_path:
            coco_images = self.download_coco_pets(coco_path, max_images=max_per_source)
            all_collected['coco'] = coco_images
        else:
            logger.warning("COCO_PATH environment variable not set. Skipping COCO collection.")
            all_collected['coco'] = []
        
        # OpenImages
        oid_dir = self.output_dir / 'openimages_toolkit'
        openimages_images = self.download_openimages_pets(oid_dir, max_images=max_per_source)
        all_collected['openimages'] = openimages_images
        
        # Summary
        total_images = sum(len(imgs) for imgs in all_collected.values())
        logger.info(f"Collected a total of {total_images} images from all sources")
        
        return all_collected


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect additional pet images from various sources')
    parser.add_argument('--output_dir', type=str, default='data/additional', 
                        help='Directory to save collected data')
    parser.add_argument('--max_per_source', type=int, default=500,
                        help='Maximum images to collect from each source')
    parser.add_argument('--flickr_key', type=str, required=True,
                        help='Flickr API key')
    parser.add_argument('--flickr_secret', type=str, required=True,
                        help='Flickr API secret')
    
    args = parser.parse_args()
    
    collector = DataCollector(
        output_dir=args.output_dir,
        flickr_api_key=args.flickr_key,
        flickr_api_secret=args.flickr_secret
    )
    
    collected_data = collector.collect_all_sources(max_per_source=args.max_per_source)
    
    # Output summary
    for source, images in collected_data.items():
        print(f"Collected {len(images)} images from {source}")
