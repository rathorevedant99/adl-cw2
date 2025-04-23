# Weakly-Supervised Semantic Segmentation: Implementation Details

This document provides an in-depth explanation of the weakly-supervised semantic segmentation approach implemented in this framework. It aims to help new users understand the key concepts, implementation details, and how the different components work together.

## Introduction to Weakly-Supervised Segmentation

### The Challenge

Fully-supervised semantic segmentation requires pixel-level annotations, which are:
- Time-consuming to create
- Expensive to obtain
- Require expert knowledge in many domains

Weakly-supervised approaches aim to achieve similar segmentation quality using only image-level labels (classification labels), which are:
- Much easier to obtain
- Often already available in existing datasets
- Can be collected at scale

### Our Approach

Our implementation uses a combination of Class Activation Maps (CAMs) and region growing to generate segmentation masks from image-level labels only.

## Technical Implementation

### 1. Model Architecture

Our model architecture consists of:

#### Backbone Network
- Can be a UNet or ResNet50 architecture
- Processes input images to extract features
- Preserves spatial information throughout the network

#### Classification Branch
- Takes features from the backbone
- Applies Global Average Pooling (GAP)
- Produces class predictions with a final linear layer

#### Segmentation Branch
- Uses the same backbone features
- Generates Class Activation Maps (CAMs)
- Optionally applies region growing to refine the segmentation

```
Input Image
    │
    ▼
┌───────────┐
│  Backbone │
└───────────┘
    │
    ▼
┌───────────────────┬───────────────────┐
│ Classification    │ Segmentation      │
│ Branch            │ Branch            │
├───────────────────┼───────────────────┤
│ Global Average    │ Class Activation  │
│ Pooling           │ Maps              │
│                   │                   │
│ Linear Layer      │ Region Growing    │
│                   │ (optional)        │
└───────────────────┴───────────────────┘
    │                   │
    ▼                   ▼
┌───────────┐      ┌───────────┐
│  Class    │      │  Segmen-  │
│  Predi-   │      │  tation   │
│  ctions   │      │  Masks    │
└───────────┘      └───────────┘
```

### 2. Forward Pass

#### UNet Implementation (in `WeaklySupervisedSegmentationModelUNet`)

```python
def forward(self, x, labels=None, apply_region_growing=False):
    # Forward through UNet
    cam = self.unet(x)  # [B, C, H, W]
    
    # Apply ReLU activation to get positive CAMs
    cam = F.relu(cam)

    # Classification branch: Global Average Pooling
    pooled = self.gap(cam).view(cam.size(0), -1)

    # Segmentation maps are initially just the CAMs
    segmentation_maps = cam

    # Apply region growing if requested
    if apply_region_growing or labels is not None:
        feature_map = self.feature_proj(cam)
        segmentation_maps = self._region_growing(segmentation_maps, feature_map, labels)

    return {
        'logits': pooled,  # For classification loss
        'segmentation_maps': segmentation_maps,  # For segmentation loss and visualization
    }
```

### 3. Loss Functions

Our training uses two loss components:

#### Classification Loss
```python
# Classification loss using cross-entropy
cls_loss = self.cls_criterion(logits, labels)
```

#### Weakly-Supervised Segmentation Loss
This is a Dice-based loss that encourages the CAM for the correct class to be coherent:

```python
def _calculate_weak_supervision_loss(self, segmentation_maps, labels):
    # For each image in the batch
    for i in range(batch_size):
        # Get the CAM for the correct class
        seg_map = segmentation_maps[i]
        label_idx = labels[i].item()
        target_activation = seg_map[label_idx]
        
        # Apply sigmoid to scale activations
        target_activation = torch.sigmoid(target_activation)

        # Create binary mask using threshold
        threshold = 0.5
        binary_mask = (target_activation > threshold).float()

        # Calculate Dice score between activation and binary mask
        intersection = (target_activation * binary_mask).sum()
        union = target_activation.sum() + binary_mask.sum()
        dice_score = (2 * intersection + 1e-6) / (union + 1e-6)
        
        # Loss is 1 - dice_score
        dice_loss = 1 - dice_score
        loss += dice_loss
```

This loss encourages:
1. High activation values in regions that exceed the threshold
2. Low activation values in regions below the threshold
3. Spatial consistency in the CAM

#### Total Loss
```python
# Total loss is a weighted sum
loss = cls_loss + self.seg_loss_weight * seg_loss
```

The `seg_loss_weight` parameter (default: 0.1) balances the classification and segmentation objectives.

### 4. Region Growing

The region growing algorithm refines the initial CAMs to produce better segmentation masks. It operates in feature space to find regions with similar features to the initial segmentation.

Key steps in the algorithm:

1. **Initialize with thresholded CAMs**:
   ```python
   refined_masks = (cam_maps > self.cam_threshold).float()
   ```

2. **Normalize feature maps**:
   ```python
   norm = torch.norm(feature_map, dim=1, keepdim=True)
   norm = torch.clamp(norm, min=1e-8)
   normed_features = feature_map / norm
   ```

3. **Iterative region growing**:
   For each foreground region:
   - Calculate mean feature of current foreground
   - Calculate similarity of each pixel to foreground mean
   - Add adjacent pixels with high similarity to foreground
   - Update masks with new pixels and confidence scores

4. **Finalize masks**:
   ```python
   # Normalize to [0,1]
   refined_masks = torch.clamp(refined_masks, 0.0, 1.0)
   
   # One-hot encoding based on class with maximum probability
   max_vals, max_inds = torch.max(refined_masks, dim=1, keepdim=True)
   one_hot = torch.zeros_like(refined_masks)
   one_hot.scatter_(1, max_inds, 1.0)
   
   # Soften the one-hot masks
   softened = refined_masks * one_hot
   ```

### 5. Training Process

The training process for weakly-supervised segmentation:

1. Load a batch of images and their class labels
2. Forward pass through the model to get class predictions and CAMs
3. Calculate classification loss
4. Calculate segmentation loss using the CAMs
5. Combine losses and backpropagate
6. Update model parameters
7. Periodically evaluate on validation set
8. Save checkpoints

## Practical Considerations

### When to Use Region Growing

Region growing provides more refined segmentation masks but adds computational overhead. It's recommended to:

- Set `apply_region_growing: false` during training for faster iterations
- Enable region growing during evaluation for better segmentation quality
- Experiment with both options if you have computation resources available

### Hyperparameters to Tune

For optimal results, consider tuning these hyperparameters:

- `seg_loss_weight`: Controls the influence of the segmentation loss (default: 0.1)
- `cam_threshold`: Threshold for initial CAM binarization (default: 0.1)
- `region_growing_iterations`: Number of region growing iterations (default: 5)

### Debugging and Visualization

To understand how the model is performing:

1. Check the CAM visualizations in `experiments/visualizations/cams/`
2. Compare ground truth masks with predictions in `experiments/visualizations/`
3. Monitor classification accuracy alongside mean IoU

## Comparing with Fully-Supervised Approach

Our framework allows direct comparison between weakly-supervised and fully-supervised approaches:

1. Train both models with the same backbone and hyperparameters
2. Compare Mean IoU and Pixel Accuracy
3. Analyze where weakly-supervised method struggles (typically with fine details)

## Extending the Framework

To adapt this approach to your own dataset:

1. Create a dataset class similar to `PetDataset`
2. Ensure it can return both image-level labels and pixel masks
3. Adjust the number of classes in the configuration
4. Consider domain-specific augmentations

## References and Further Reading

For a deeper understanding of the techniques used:

1. Zhou, B., et al. (2016). "Learning Deep Features for Discriminative Localization." CVPR.
2. Ahn, J., & Kwak, S. (2018). "Learning Pixel-level Semantic Affinity with Image-level Supervision for Weakly Supervised Semantic Segmentation." CVPR.
3. Wei, Y., et al. (2017). "Object Region Mining with Adversarial Erasing: A Simple Classification to Semantic Segmentation Approach." CVPR. 