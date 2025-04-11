import torch.nn.functional as F
import torch

def iou_loss(pred, target, smooth=1.0):
    """
    Calculate IoU-based loss (1 - IoU)
    """
    # Flatten predictions and targets
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    # Calculate intersection and union
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    
    # Calculate IoU with smoothing term to avoid division by zero
    iou = (intersection + smooth) / (union + smooth)
    
    # Return 1 - IoU as the loss (lower is better)
    return 1.0 - iou

def generator_loss_fn(pred_fake, pred_mask, real_mask, lambda_adv=1.0, lambda_seg=100, lambda_iou=50):
    """
    pred_fake: Discriminator output for the fake pair (image + generated_mask).
    pred_mask: The generated mask from the generator.
    real_mask: The ground-truth mask.
    lambda_adv: Weight for adversarial loss (reduced to prevent domination)
    lambda_seg: Weight for BCE segmentation loss.
    lambda_iou: Weight for IoU loss.
    """
    # Adversarial loss: want pred_fake to be "real" -> target=1
    adv_loss = F.binary_cross_entropy_with_logits(pred_fake, torch.ones_like(pred_fake))
    
    # Ensure pred_mask has values in proper range to avoid NaN in BCE
    # This adds numerical stability
    pred_mask_safe = torch.clamp(pred_mask, 1e-7, 1.0 - 1e-7)
    
    # BCE segmentation loss
    bce_loss = F.binary_cross_entropy(pred_mask_safe, real_mask)
    
    # IoU loss
    iou = iou_loss(pred_mask, real_mask)
    
    # Combine losses with weights
    total_loss = lambda_adv * adv_loss + lambda_seg * bce_loss + lambda_iou * iou
    
    # Return both total loss and individual components for logging
    return total_loss, adv_loss, bce_loss, iou

def discriminator_loss_fn(pred_real, pred_fake):
    """
    pred_real: D output on real pair (image + real_mask).
    pred_fake: D output on fake pair (image + generated_mask).
    """
    # Real pairs -> label = 1 (with label smoothing for stability)
    real_labels = torch.ones_like(pred_real) * 0.9  # Soft label = 0.9 instead of 1
    
    # Fake pairs -> label = 0
    fake_labels = torch.zeros_like(pred_fake)
    
    # Calculate losses
    real_loss = F.binary_cross_entropy_with_logits(pred_real, real_labels)
    fake_loss = F.binary_cross_entropy_with_logits(pred_fake, fake_labels)
    
    return 0.5 * (real_loss + fake_loss)