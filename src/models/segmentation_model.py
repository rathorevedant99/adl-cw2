import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

class WeaklySupervisedSegmentationModel(nn.Module):
    def __init__(self, num_classes, backbone='resnet50'):
        super().__init__()
        
        # Initialize backbone
        if backbone == 'resnet50':
            self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
            # Remove the final fully connected layer
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
            backbone_channels = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Class-specific feature maps
        self.class_specific = nn.Conv2d(backbone_channels, num_classes, kernel_size=1)
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Generate class activation maps
        cam = self.class_specific(features)
        
        # Global average pooling for classification
        pooled = self.gap(cam)
        pooled = pooled.view(pooled.size(0), -1)
        
        # Generate segmentation maps
        segmentation_maps = F.interpolate(
            cam,
            size=x.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        
        return {
            'logits': pooled,
            'segmentation_maps': segmentation_maps
        }
    
    def get_cam_maps(self, x, target_class):
        """Get Class Activation Maps for a specific class"""
        features = self.backbone(x)
        cam = self.class_specific(features)
        
        # Get CAM for target class
        cam = cam[:, target_class:target_class+1]
        
        # Normalize CAM
        cam = F.relu(cam)
        cam = cam / (cam.max() + 1e-8)
        
        return cam 