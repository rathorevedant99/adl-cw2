import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
class WeaklySupervisedSegmentationModel(nn.Module):
    """
    Weakly Supervised Segmentation Model
    """
    def __init__(self, num_classes, backbone='resnet50'):
        super().__init__()
        
        # Initialize backbone
        if backbone == 'resnet50':
            self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-2]) # Remove the final fully connected layer as we just want the features
            backbone_channels = 2048 # The number of channels in the last convolutional layer of the backbone
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        self.class_specific = nn.Conv2d(backbone_channels, num_classes, kernel_size=1) # A convolutional layer that outputs a feature map for each class
        
        self.gap = nn.AdaptiveAvgPool2d(1) # A global average pooling layer that reduces the spatial dimensions of the feature maps to 1x1
        
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
        
        # Apply ReLU to ensure non-negative values
        cam = F.relu(cam)
        
        # For classification logits
        pooled = self.gap(cam) # Applies global average pooling to the class activation maps
        pooled = pooled.view(pooled.size(0), -1) # Flattens the class activation maps
        
        # Generate segmentation maps with proper normalization
        segmentation_maps = F.interpolate(
            cam,
            size=x.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        
        # Apply softmax to get proper probability distributions
        # Reshape for softmax: [B, C, H, W] -> [B, H*W, C] -> softmax -> [B, H*W, C] -> [B, C, H, W]
        B, C, H, W = segmentation_maps.shape
        segmentation_maps = segmentation_maps.permute(0, 2, 3, 1).contiguous().view(B, H*W, C)
        segmentation_maps = F.softmax(segmentation_maps, dim=2)
        segmentation_maps = segmentation_maps.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        
        return {
            'logits': pooled,
            'segmentation_maps': segmentation_maps
        }
    
    def get_cam_maps(self, x, target_class):
        """Get Class Activation Maps for a specific class"""
        features = self.backbone(x)
        cam = self.class_specific(features)
        
        cam = cam[:, target_class:target_class+1] # Extracts the class activation map for the target class
        
        cam = F.relu(cam)
        cam = cam / (cam.max() + 1e-8) # Normalizes the class activation map
        
        return cam 