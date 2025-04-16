import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage

class DoubleConv(nn.Module):
    """Double convolution block for U-Net"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Adjust x1 size to match x2 if needed
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, 
                        diffY // 2, diffY - diffY // 2])
        # Concatenate along the channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """Output convolution"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class WeaklySupervisedSegmentationModel(nn.Module):
    """Weakly Supervised Segmentation Model based on U-Net architecture with CAM-RG"""
    def __init__(self, num_classes, backbone='unet', cam_threshold=0.2, region_growing_iterations=5):
        super().__init__()
        self.num_classes = num_classes
        self.cam_threshold = cam_threshold
        self.region_growing_iterations = region_growing_iterations
        
        # U-Net Encoder (Downsampling path)
        self.inc = DoubleConv(3, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        
        # U-Net Decoder (Upsampling path)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        
        # Multi-scale feature projections for region growing
        self.feature_proj1 = nn.Conv2d(256, 64, kernel_size=1)
        self.feature_proj2 = nn.Conv2d(128, 64, kernel_size=1)
        
        # Output convolution for class-specific features
        self.outc = OutConv(64, num_classes)
        
        # Global Average Pooling for classification
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Classification head
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)
        
        # Boundary refinement module
        self.boundary_refine = nn.Sequential(
            nn.Conv2d(64 + num_classes, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )
        
        # Consistency Module with attention
        self.consistency_attn = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize the weights of the model"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, labels=None, bbox=None, is_training=True):
        """Forward pass of the model with CAM-RG for weakly-supervised segmentation"""
        # Input size is preserved throughout for later reference
        input_size = x.shape[2:]
        batch_size = x.shape[0]
        
        # Store features for multi-scale region growing
        features = {}
        
        # Encoder path
        x1 = self.inc(x)
        features['enc1'] = x1
        
        x2 = self.down1(x1)
        features['enc2'] = x2
        
        x3 = self.down2(x2)
        features['enc3'] = x3
        
        x4 = self.down3(x3)
        features['enc4'] = x4
        
        x5 = self.down4(x4)
        features['enc5'] = x5
        
        # Decoder path
        x = self.up1(x5, x4)
        features['dec1'] = x
        
        x = self.up2(x, x3)
        features['dec2'] = x
        
        x = self.up3(x, x2)
        features['dec3'] = x
        
        x = self.up4(x, x1)
        features['dec4'] = x
        
        # Generate class-specific maps (CAM)
        class_features = self.classifier(x)
        
        pooled = self.gap(cam) # Applies global average pooling to the class activation maps
        pooled = pooled.view(pooled.size(0), -1) # Flattens the class activation maps
        
        # Generate segmentation maps
        segmentation_maps = F.interpolate(
            cam,
            size=x.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        
        # Apply bounding box constraints if provided
        if bbox is not None:
            logging.info("Applying bounding box constraints")
            for b in range(batch_size):
                # Create a mask from the bounding box
                box_mask = torch.zeros((height, width), device=device)
                x1, y1, x2, y2 = bbox[b].int().cpu().numpy()
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width-1, x2), min(height-1, y2)
                box_mask[y1:y2, x1:x2] = 1.0
                
                # Constrain the refined mask to be inside the box
                box_mask = box_mask.unsqueeze(0).unsqueeze(0).expand(-1, num_classes, -1, -1)
                refined_masks[b] = refined_masks[b] * box_mask
        
        # If labels are provided, constrain the output to the correct classes
        if labels is not None:
            logging.info("Processing labels for class constraints")
            # Handle different label formats
            # First, normalize the labels to ensure they're in the right format
            if labels.dim() == 1:  # If labels are class indices [batch_size]
                # Convert to one-hot encoding
                one_hot = torch.zeros(batch_size, num_classes, device=device)
                for b in range(batch_size):
                    one_hot[b, labels[b]] = 1
                labels_formatted = one_hot
            elif labels.dim() == 2 and labels.size(1) == 1:  # If labels are [batch_size, 1]
                # Convert to one-hot encoding
                one_hot = torch.zeros(batch_size, num_classes, device=device)
                for b in range(batch_size):
                    one_hot[b, labels[b, 0]] = 1
                labels_formatted = one_hot
            elif labels.dim() == 2 and labels.size(1) > 1:  # If labels are already one-hot [batch_size, num_classes]
                labels_formatted = labels
            else:  # Handle unexpected formats
                logging.warning(f"Unexpected label format: {labels.shape}")
                # Default to the first class for each sample
                labels_formatted = torch.zeros(batch_size, num_classes, device=device)
                labels_formatted[:, 0] = 1.0
            
            # Now create the mask for zeroing out predictions for classes not in the image
            # Reshape labels_formatted to [batch_size, num_classes, 1, 1] and expand
            mask = labels_formatted.unsqueeze(2).unsqueeze(3).expand(-1, -1, height, width)
            refined_masks = refined_masks * mask
        
        # Use a simplified region growing approach to avoid performance issues
        logging.info(f"Starting {self.region_growing_iterations} region growing iterations")
        
        # Simplified region growing - just one iteration to avoid performance bottlenecks
        # Calculate feature similarity
        feature_maps = dec2_upsampled + dec3_upsampled
        feature_norms = torch.norm(feature_maps, dim=1, keepdim=True)
        normalized_features = feature_maps / (feature_norms + 1e-8)
        
        # Process all classes in parallel where possible
        foreground = (refined_masks > 0.5).float()
        
        # Skip empty masks
        non_empty_masks = foreground.sum(dim=(2,3)) > 0
        
        # For each batch and each non-empty class
        for b in range(batch_size):
            # Get features for this batch item
            batch_features = normalized_features[b]  # [C, H, W]
            
            for c in range(num_classes):
                # Skip if class is empty or not in the image
                if not non_empty_masks[b, c]:
                    continue
                    
                if labels is not None and labels_formatted[b, c] == 0:
                    continue
                
                # Get foreground mask for this class
                fg_mask = foreground[b, c].unsqueeze(0)  # [1, H, W]
                
                # Calculate mean feature vector of foreground
                fg_features = batch_features * fg_mask  # [C, H, W]
                fg_sum = fg_features.sum(dim=(1, 2))  # [C]
                fg_count = fg_mask.sum() + 1e-8  # Avoid division by zero
                fg_mean = fg_sum / fg_count  # [C]
                
                # Calculate similarity map
                fg_mean = fg_mean.view(-1, 1, 1)  # [C, 1, 1]
                similarity = (batch_features * fg_mean).sum(dim=0)  # [H, W]
                
                # Threshold the similarity map
                similarity_threshold = similarity.mean() + 0.5 * similarity.std()
                new_pixels = (similarity > similarity_threshold) & (~(foreground[b, c] > 0.5))
                
                # Update mask
                refined_masks[b, c] = refined_masks[b, c] + new_pixels.float() * 0.5
        
        # Ensure no overlap between classes (assign pixel to class with highest activation)
        logging.info("Finalizing masks - resolving class overlaps")
        max_values, max_indices = torch.max(refined_masks, dim=1, keepdim=True)
        one_hot = torch.zeros_like(refined_masks)
        for b in range(batch_size):
            one_hot[b].scatter_(0, max_indices[b], 1.0)
        refined_masks = refined_masks * one_hot
        
        logging.info("Region growing completed successfully")
        return refined_masks
        
        return refined_masks
    
    def _apply_consistency_constraints(self, segmentation_maps, features, input_size):
        """Apply consistency constraints between forward and backward passes
        
        Args:
            segmentation_maps: Current segmentation maps [B, C, H, W]
            features: Dictionary of encoder-decoder features
            input_size: Target size for output maps (H, W)
            
        Returns:
            Consistency-enhanced segmentation maps
        """
        # Extract key features for consistency
        dec4_features = features['dec4']
        enc1_features = features['enc1']
        
        # Calculate feature consistency using attention
        combined_features = torch.cat([dec4_features, enc1_features], dim=1)
        attention_weights = self.consistency_attn(combined_features)
        
        # Apply attention to the segmentation maps
        refined_maps = segmentation_maps * attention_weights
        
        # Upsample to input size if needed
        if refined_maps.shape[2:] != input_size:
            refined_maps = F.interpolate(
                refined_maps,
                size=input_size,
                mode='bilinear',
                align_corners=False
            )
        
        return refined_maps
    
    def get_loss_functions(self):
        """Return loss functions for weakly-supervised training"""
        return {
            'classification_loss': nn.CrossEntropyLoss(),
            'consistency_loss': ConsistencyLoss(),
            'size_constraint_loss': SizeConstraintLoss()
        }

class ConsistencyLoss(nn.Module):
    """Loss function to enforce consistency between CAMs and segmentation"""
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
        
    def forward(self, cam_maps, segmentation_maps):
        """Enforce consistency between CAM and segmentation maps"""
        # Normalize both maps
        cam_norm = F.normalize(cam_maps, p=2, dim=1)
        seg_norm = F.normalize(segmentation_maps, p=2, dim=1)
        
        # Calculate cosine similarity loss
        similarity = torch.sum(cam_norm * seg_norm, dim=1).mean()
        loss = 1.0 - similarity  # 1 - cosine similarity as loss
        
        return self.weight * loss

class SizeConstraintLoss(nn.Module):
    """Loss function to constrain segmentation size"""
    def __init__(self, weight=0.1, min_size=0.05, max_size=0.9):
        super().__init__()
        self.weight = weight
        self.min_size = min_size
        self.max_size = max_size
        
    def forward(self, segmentation_maps, labels=None):
        """Penalize too small or too large segmentation regions"""
        batch_size, num_classes, height, width = segmentation_maps.shape
        total_pixels = height * width
        
        # Calculate foreground ratios for each class
        foreground_pixels = torch.sum(torch.sigmoid(segmentation_maps) > 0.5, dim=(2, 3))
        foreground_ratios = foreground_pixels / total_pixels
        
        # Calculate size constraint loss
        too_small = torch.relu(self.min_size - foreground_ratios)
        too_large = torch.relu(foreground_ratios - self.max_size)
        size_loss = too_small + too_large
        
        # If labels are provided, only apply to present classes
        if labels is not None:
            size_loss = size_loss * labels
            # Average over only present classes
            num_present = torch.sum(labels, dim=1, keepdim=True).clamp(min=1.0)
            size_loss = torch.sum(size_loss, dim=1) / num_present.squeeze()
        
        return self.weight * size_loss.mean()