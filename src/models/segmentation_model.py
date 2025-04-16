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

        for m in self.modules(): # For each module in the model
            if isinstance(m, nn.Conv2d): # If the module is a convolutional layer
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') # Initialize the weights using kaiming_normal initialization
                if m.bias is not None: # If the bias is not None
                    nn.init.constant_(m.bias, 0) # Initialize the bias to 0
    
    def forward(self, x):
        """Forward pass of the model"""

        features = self.backbone(x) # Extracts the features using the backbone we defined
        
        cam = self.class_specific(features) # Generates the class activation maps
        
        pooled = self.gap(cam) # Applies global average pooling to the class activation maps
        pooled = pooled.view(pooled.size(0), -1) # Flattens the class activation maps
        
        # Generate segmentation maps
        segmentation_maps = F.interpolate(
            cam,
            size=x.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        
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