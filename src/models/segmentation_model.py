import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from unet import UNet
# ---- 2.  Wrap it in your weakly-supervised scaffold ----- #
class WeaklySupervisedUNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.unet = UNet(num_classes=num_classes)
        # 1¡Á1 conv produces CAMs from the *bottleneck* feature map
        self.class_specific = nn.Conv2d(512, num_classes, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        seg, bottleneck = self.unet(x)      # full?res seg, low?res feats
        cam = F.relu(self.class_specific(bottleneck))
        logits = self.gap(cam).flatten(1)

        cam_up = F.interpolate(cam, size=x.shape[2:], mode='bilinear',
                               align_corners=False)

        return {"logits": logits,
                "segmentation_maps": seg,
                "cam_maps": cam_up}

    def get_cam_maps(self, x, target_class):
        _, bottleneck = self.unet(x)
        cam = F.relu(self.class_specific(bottleneck))[:, target_class:target_class+1]
        cam = cam / (cam.max() + 1e-8)
        return F.interpolate(cam, size=x.shape[2:], mode='bilinear',
                             align_corners=False)
