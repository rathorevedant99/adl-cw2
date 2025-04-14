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
        
        # Apply ReLU to ensure non-negative values
        cam = F.relu(class_features)
        
        # For classification logits
        pooled = self.gap(cam)
        logits = pooled.view(pooled.size(0), -1)
        
        # Apply region growing if in training mode or explicitly requested
        if is_training or labels is not None:
            # Generate initial segmentation maps from CAM
            cam_upsampled = F.interpolate(
                cam,
                size=input_size,
                mode='bilinear',
                align_corners=False
            )
            
            # Apply region growing algorithm
            refined_maps = self._region_growing(cam_upsampled, features, labels, bbox)
            
            # Apply consistency constraints
            consistency_maps = self._apply_consistency_constraints(refined_maps, features, input_size)
            
            # Boundary refinement using encoder-decoder features
            final_features = torch.cat([x, consistency_maps], dim=1)
            refined_segmentation = self.boundary_refine(final_features)
            
            segmentation_maps = F.interpolate(
                refined_segmentation,
                size=input_size,
                mode='bilinear',
                align_corners=False
            )
        else:
            # For inference, just use the segmentation output
            segmentation_maps = self.outc(x)
            segmentation_maps = F.interpolate(
                segmentation_maps,
                size=input_size,
                mode='bilinear',
                align_corners=False
            )
        
        outputs = {
            'logits': logits,
            'segmentation_maps': segmentation_maps,
            'cam_maps': cam
        }
        
        # Add multi-scale features and intermediate results if in training mode
        if is_training:
            outputs['features'] = features
        
        return outputs
    
    def get_cam_maps(self, x, target_class, apply_region_growing=False):
        """Get Class Activation Maps for a specific class with optional region growing"""
        # Store features for region growing if needed
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
        
        # Generate class-specific maps
        cam = self.classifier(x)
        
        # Select target class
        cam = cam[:, target_class:target_class+1]
        
        # Apply ReLU and normalize
        cam = F.relu(cam)
        cam = cam / (cam.max() + 1e-8)
        
        # Upsample to input size
        input_size = x1.shape[2:]
        cam_upsampled = F.interpolate(
            cam,
            size=input_size,
            mode='bilinear',
            align_corners=False
        )
        
        if apply_region_growing:
            # Create a one-hot label for the target class
            labels = torch.zeros(x.shape[0], self.num_classes, device=x.device)
            labels[:, target_class] = 1.0
            
            # Apply region growing to refine the CAM
            refined_cam = self._region_growing(cam_upsampled, features, labels)
            return refined_cam
        
        return cam_upsampled
        
    def _region_growing(self, cam_maps, features, labels=None, bbox=None):
        """Apply region growing algorithm to refine CAM maps
        
        Args:
            cam_maps: Initial CAM maps [B, C, H, W]
            features: Dictionary of encoder-decoder features
            labels: One-hot encoded class labels [B, C]
            bbox: Optional bounding box coordinates [B, 4] (x1, y1, x2, y2)
            
        Returns:
            Refined segmentation maps
        """
        batch_size, num_classes, height, width = cam_maps.shape
        device = cam_maps.device
        
        # Extract multi-scale features for region growing
        dec2_features = self.feature_proj1(features['dec2'])
        dec3_features = self.feature_proj2(features['dec3'])
        
        # Upsample all features to the same resolution
        dec2_upsampled = F.interpolate(dec2_features, size=(height, width), mode='bilinear', align_corners=False)
        dec3_upsampled = F.interpolate(dec3_features, size=(height, width), mode='bilinear', align_corners=False)
        
        # Initialize masks with thresholded CAM
        refined_masks = (cam_maps > self.cam_threshold).float()
        
        # Apply bounding box constraints if provided
        if bbox is not None:
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
                # Try to reshape if possible, otherwise use as is
                try:
                    labels_formatted = labels.view(batch_size, -1)
                    if labels_formatted.size(1) != num_classes:
                        # If not matching num_classes, use argmax to get class indices
                        class_indices = torch.argmax(labels_formatted, dim=1)
                        one_hot = torch.zeros(batch_size, num_classes, device=device)
                        for b in range(batch_size):
                            one_hot[b, class_indices[b]] = 1
                        labels_formatted = one_hot
                except:
                    # If reshape fails, just use the first batch dimension
                    if labels.size(0) == batch_size:
                        class_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
                        one_hot = torch.zeros(batch_size, num_classes, device=device)
                        for b in range(batch_size):
                            one_hot[b, class_indices[b]] = 1
                        labels_formatted = one_hot
            
            # Now create the mask for zeroing out predictions for classes not in the image
            # Reshape labels_formatted to [batch_size, num_classes, 1, 1] and expand
            mask = labels_formatted.unsqueeze(2).unsqueeze(3).expand(-1, -1, height, width)
            refined_masks = refined_masks * mask
        
        # Region growing iterations
        for _ in range(self.region_growing_iterations):
            # Calculate feature similarity
            feature_maps = dec2_upsampled + dec3_upsampled
            feature_norms = torch.norm(feature_maps, dim=1, keepdim=True)
            normalized_features = feature_maps / (feature_norms + 1e-8)
            
            # For each class, grow regions based on feature similarity
            for c in range(num_classes):
                # Skip empty classes
                if labels is not None:
                    # Use the formatted labels we created earlier
                    if not labels_formatted[:, c].any():
                        continue
                
                for b in range(batch_size):
                    # Skip if this class isn't present in this image
                    if labels is not None and labels_formatted[b, c] == 0:
                        continue
                    
                    # Convert to numpy for scipy operations
                    mask_np = refined_masks[b, c].detach().cpu().numpy()
                    
                    # Find boundaries of current mask
                    # This calculates a binary edge map of the mask
                    struct = ndimage.generate_binary_structure(2, 2)  # 8-connectivity
                    boundary = mask_np - ndimage.binary_erosion(mask_np, structure=struct).astype(mask_np.dtype)
                    boundary_indices = np.where(boundary > 0)
                    
                    if len(boundary_indices[0]) == 0:
                        continue  # No boundary, skip
                    
                    # Get feature vectors for boundary pixels
                    boundary_features = normalized_features[b, :, boundary_indices[0], boundary_indices[1]]
                    
                    # Calculate similarity between boundary and neighboring pixels
                    # Create a dilated version of the boundary
                    dilated = ndimage.binary_dilation(boundary, structure=struct).astype(np.bool_)
                    candidate_indices = np.where(np.logical_and(dilated, ~mask_np.astype(np.bool_)))
                    
                    if len(candidate_indices[0]) == 0:
                        continue  # No candidates, skip
                    
                    # Get feature vectors for candidate pixels
                    candidate_features = normalized_features[b, :, candidate_indices[0], candidate_indices[1]]
                    
                    # Find nearest boundary point for each candidate
                    similarities = torch.zeros((len(candidate_indices[0]),), device=device)
                    
                    # This is a simplified approach that checks similarity to the average boundary feature
                    avg_boundary_feature = boundary_features.mean(dim=1, keepdim=True)
                    for idx in range(len(candidate_indices[0])):
                        candidate_feat = candidate_features[:, idx:idx+1]
                        similarity = torch.cosine_similarity(avg_boundary_feature, candidate_feat, dim=0)
                        similarities[idx] = similarity
                    
                    # Create a mask of pixels to add (those with high similarity)
                    to_add = similarities > 0.85  # Similarity threshold
                    
                    if to_add.any():
                        # Update the mask with new pixels
                        add_y = candidate_indices[0][to_add.cpu().numpy()]
                        add_x = candidate_indices[1][to_add.cpu().numpy()]
                        mask_np[add_y, add_x] = 1
                        refined_masks[b, c] = torch.from_numpy(mask_np).to(device)
            
            # Ensure no overlap between classes (assign pixel to class with highest CAM value)
            max_values, max_indices = torch.max(cam_maps * refined_masks, dim=1, keepdim=True)
            one_hot = torch.zeros_like(refined_masks)
            for b in range(batch_size):
                one_hot[b].scatter_(0, max_indices[b], 1.0)
            refined_masks = refined_masks * one_hot
        
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