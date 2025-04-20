import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet import UNetGenerator

class WeaklySupervisedSegmentationModelUNet(nn.Module):
    def __init__(self, num_classes, cam_threshold=0.1, region_growing_iterations=5):
        super().__init__()
        self.num_classes = num_classes
        self.cam_threshold = cam_threshold
        self.region_growing_iterations = region_growing_iterations

        # Use UNetGenerator as backbone
        self.unet = UNetGenerator(in_channels=3, out_channels=num_classes)

        # Feature projection for region growing
        self.feature_proj = nn.Conv2d(num_classes, 64, kernel_size=1)  # Project CAM output for region growing

        # Classification head (GAP on output CAMs)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, labels=None, apply_region_growing=False):
        cam = self.unet(x)  # [B, C, H, W]
        cam = F.relu(cam)

        # Classification logits
        pooled = self.gap(cam).view(cam.size(0), -1)

        segmentation_maps = cam

        if apply_region_growing or labels is not None:
            feature_map = self.feature_proj(cam)  # projection of CAM output
            segmentation_maps = self._region_growing(segmentation_maps, feature_map, labels)

        return {
            'logits': pooled,
            'segmentation_maps': segmentation_maps,
        }

    def _region_growing(self, cam_maps, feature_map, labels=None):
        batch_size, num_classes, H, W = cam_maps.shape

        # Initialize refined masks from CAM thresholding
        refined_masks = (cam_maps > self.cam_threshold).float()

        # Normalize features
        norm = torch.norm(feature_map, dim=1, keepdim=True)
        normed_features = feature_map / (norm + 1e-8)

        for iteration in range(self.region_growing_iterations):
            foreground = (refined_masks > 0.5).float()
            non_empty = foreground.sum(dim=(2, 3)) > 0
            new_additions = torch.zeros_like(refined_masks)

            for b in range(batch_size):
                for c in range(num_classes):
                    if not non_empty[b, c]:
                        continue
                    if labels is not None and labels[b, c] == 0:
                        continue

                    fg_mask = foreground[b, c].unsqueeze(0)
                    fg_feat = normed_features[b] * fg_mask

                    fg_sum = fg_feat.sum(dim=(1, 2))
                    fg_count = fg_mask.sum() + 1e-8
                    fg_mean = fg_sum / fg_count
                    fg_mean = fg_mean.view(-1, 1, 1)

                    similarity = (normed_features[b] * fg_mean).sum(dim=0)

                    strictness_factor = 0.5 - 0.05 * iteration
                    threshold = similarity.mean() + strictness_factor * similarity.std()

                    if iteration > 0:
                        kernel = torch.ones(1, 1, 3, 3, device=foreground.device)
                        dilated = F.conv2d(fg_mask.unsqueeze(0), kernel, padding=1)[0, 0] > 0
                        new_pixels = (similarity > threshold) & (~(foreground[b, c] > 0.5)) & dilated
                    else:
                        new_pixels = (similarity > threshold) & (~(foreground[b, c] > 0.5))

                    confidence = (similarity[new_pixels] - threshold) / (similarity.max() - threshold + 1e-8)
                    confidence = torch.clamp(confidence, 0.1, 0.9)

                    temp_mask = torch.zeros_like(refined_masks[b, c])
                    temp_mask[new_pixels] = confidence.float()
                    new_additions[b, c] = temp_mask

            refined_masks = refined_masks + new_additions

        refined_masks = torch.clamp(refined_masks, 0.0, 1.0)
        max_vals, max_inds = torch.max(refined_masks, dim=1, keepdim=True)
        one_hot = torch.zeros_like(refined_masks)
        one_hot.scatter_(1, max_inds, 1.0)
        softened = refined_masks * one_hot
        return softened

class FullySupervisedSegmentationModelUNet(nn.Module):
    def __init__(self, num_classes, base_channels=32, bilinear=True):
        super().__init__()
        # UNetGenerator(in, out) already builds the encoder–decoder
        self.unet = UNetGenerator(in_channels=3,
                                  out_channels=num_classes,
                                  base_channels=base_channels,
                                  bilinear=bilinear)

    def forward(self, x, **kwargs):
        # ignore any extra args (labels/apply_region_growing)
        seg_maps = self.unet(x)              # [B, C=num_classes, H, W]
        return {'segmentation_maps': seg_maps}