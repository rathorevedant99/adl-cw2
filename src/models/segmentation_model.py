import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
class WeaklySupervisedSegmentationModel(nn.Module):
    def __init__(self, num_classes, backbone='resnet50', cam_threshold=0.1, region_growing_iterations=5):
        super().__init__()
        self.num_classes = num_classes
        self.cam_threshold = cam_threshold
        self.region_growing_iterations = region_growing_iterations

        if backbone == 'resnet50':
            resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
            self.layer1 = resnet.layer1
            self.layer2 = resnet.layer2
            self.layer3 = resnet.layer3  # Used for region growing features
            self.layer4 = resnet.layer4
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.class_specific = nn.Conv2d(2048, num_classes, kernel_size=1)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Feature projection for region growing
        self.feature_proj = nn.Conv2d(1024, 64, kernel_size=1)  # 1024 from layer3

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, labels=None, apply_region_growing=False):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)  # Intermediate features for region growing
        x4 = self.layer4(x3)

        cam = self.class_specific(x4)
        cam = F.relu(cam)

        pooled = self.gap(cam).view(cam.size(0), -1)

        segmentation_maps = F.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=False)

        if apply_region_growing or labels is not None:
            projected_feat = self.feature_proj(x3)
            segmentation_maps = self._region_growing(segmentation_maps, projected_feat, labels)

        return {
            'logits': pooled,
            'segmentation_maps': segmentation_maps,
        }

    def _region_growing(self, cam_maps, feature_map, labels=None):
        batch_size, num_classes, H, W = cam_maps.shape
        
        # Initialize refined masks from CAM thresholding
        refined_masks = (cam_maps > self.cam_threshold).float()
        
        # Interpolate and normalize feature maps
        feature_map = F.interpolate(feature_map, size=(H, W), mode='bilinear', align_corners=False)
        norm = torch.norm(feature_map, dim=1, keepdim=True)
        normed_features = feature_map / (norm + 1e-8)
        
        # Run for the specified number of iterations
        for iteration in range(self.region_growing_iterations):
            # Get current foreground
            foreground = (refined_masks > 0.5).float()
            non_empty = foreground.sum(dim=(2, 3)) > 0
            
            # Create a copy to record new additions for this iteration
            new_additions = torch.zeros_like(refined_masks)
            
            for b in range(batch_size):
                for c in range(num_classes):
                    # Skip empty regions or negative labels
                    if not non_empty[b, c]:
                        continue
                    if labels is not None and labels[b, c] == 0:
                        continue

                    # Get foreground features
                    fg_mask = foreground[b, c].unsqueeze(0)
                    fg_feat = normed_features[b] * fg_mask
                    
                    # Calculate foreground feature mean
                    fg_sum = fg_feat.sum(dim=(1, 2))
                    fg_count = fg_mask.sum() + 1e-8
                    fg_mean = fg_sum / fg_count
                    fg_mean = fg_mean.view(-1, 1, 1)

                    # Calculate similarity of each pixel to foreground mean
                    similarity = (normed_features[b] * fg_mean).sum(dim=0)
                    
                    # Calculate adaptive threshold - this can be tuned
                    # As iterations progress, we can make the threshold more strict
                    strictness_factor = 0.5 - 0.05 * iteration  # Gradually decrease
                    threshold = similarity.mean() + strictness_factor * similarity.std()
                    
                    # Find new pixels to add
                    # Only consider pixels adjacent to current foreground for true region growing
                    if iteration > 0:
                        # Create a dilated version of current foreground to find adjacent pixels
                        kernel = torch.ones(3, 3, device=foreground.device)
                        dilated = torch.nn.functional.conv2d(
                            fg_mask.unsqueeze(0), 
                            kernel.unsqueeze(0).unsqueeze(0),
                            padding=1
                        )[0, 0] > 0
                        # New pixels must be adjacent to current foreground
                        new_pixels = (similarity > threshold) & (~(foreground[b, c] > 0.5)) & dilated
                    else:
                        # First iteration can add any pixels above threshold
                        new_pixels = (similarity > threshold) & (~(foreground[b, c] > 0.5))
                    
                    # Add new pixels with a confidence score
                    confidence = (similarity[new_pixels] - threshold) / (similarity.max() - threshold + 1e-8)
                    confidence = torch.clamp(confidence, 0.1, 0.9)  # Limit confidence range
                    
                    temp_mask = torch.zeros_like(refined_masks[b, c])
                    temp_mask[new_pixels] = confidence.float()
                    new_additions[b, c] = temp_mask
            
            # Add new pixels to refined masks
            refined_masks = refined_masks + new_additions
            
            # Optional: log progress during training
            if hasattr(self, 'training') and self.training and iteration % 2 == 0:
                with torch.no_grad():
                    pixels_added = (new_additions > 0).float().sum().item()
                    # if pixels_added > 0:
                        # print(f"RG Iteration {iteration}: Added {pixels_added} pixels")
        
        # Normalize refined masks to [0,1] range
        refined_masks = torch.clamp(refined_masks, 0.0, 1.0)
        
        # Apply one-hot encoding based on class with maximum probability
        max_vals, max_inds = torch.max(refined_masks, dim=1, keepdim=True)
        one_hot = torch.zeros_like(refined_masks)
        one_hot.scatter_(1, max_inds, 1.0)
        
        # Alternatively, keep soft probabilities for uncertain regions
        # This is a softer approach that preserves class probabilities
        softened = refined_masks * one_hot
        
        return softened