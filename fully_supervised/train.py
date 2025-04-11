import torch
from torch.utils.data import DataLoader
from losses import generator_loss_fn, discriminator_loss_fn, iou_loss
from discriminator import PatchDiscriminator
from Unet import UNetGenerator
from dataset import OxfordPetDataset
import glob
import os
import matplotlib.pyplot as plt
import numpy as np

# Set environment variable to avoid OpenMP warning
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def compute_accuracy(pred_mask, real_mask, threshold=0.5):
    """
    Compute pixel-wise accuracy for binary segmentation.
    """
    pred_binary = (pred_mask > threshold).float()
    correct = (pred_binary == real_mask).float().sum()
    total = real_mask.numel()
    return (correct / total).item()

def compute_iou(pred_mask, real_mask, threshold=0.5):
    """
    Compute IoU for binary segmentation.
    """
    pred_binary = (pred_mask > threshold).float()
    intersection = (pred_binary * real_mask).sum().item()
    union = ((pred_binary + real_mask) > 0).float().sum().item()
    return intersection / union if union > 0 else 0.0

def visualize_predictions(generator, val_loader, device, epoch, save_dir='./visualizations'):
    """
    Visualize predictions from the generator model.
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    generator.eval()
    with torch.no_grad():
        # Just get one batch for visualization
        imgs, masks = next(iter(val_loader))
        imgs = imgs.to(device)
        masks = masks.to(device)
        
        # Generate predictions
        preds = generator(imgs)
        
        # Process batch (up to 4 images)
        batch_size = min(4, imgs.size(0))
        
        for b in range(batch_size):
            # Convert tensors to numpy for plotting
            img_np = imgs[b].cpu().permute(1, 2, 0).numpy()
            img_np = np.clip(img_np, 0, 1)
            
            # Ensure masks are 2D arrays for matplotlib
            if masks[b].dim() > 2:
                mask_np = masks[b].squeeze().cpu().numpy()
            else:
                mask_np = masks[b].cpu().numpy()
            
            # Ensure predictions are 2D arrays
            if preds[b].dim() > 2:
                pred_np = preds[b].squeeze().cpu().numpy()
            else:
                pred_np = preds[b].cpu().numpy()
            
            # Create figure
            plt.figure(figsize=(15, 5))
            
            # Plot original image
            plt.subplot(1, 3, 1)
            plt.imshow(img_np)
            plt.title("Input Image")
            plt.axis('off')
            
            # Plot ground truth mask
            plt.subplot(1, 3, 2)
            plt.imshow(mask_np, cmap='gray')
            plt.title("Ground Truth Mask")
            plt.axis('off')
            
            # Plot predicted mask
            plt.subplot(1, 3, 3)
            plt.imshow(pred_np, cmap='gray')
            plt.title("Predicted Mask")
            plt.axis('off')
            
            # Save figure
            plt.tight_layout()
            plt.savefig(f"{save_dir}/epoch_{epoch+1}_sample_{b}.png", dpi=150)
            plt.close()
    
    print(f"Saved visualization samples for epoch {epoch+1}")

def train_gan(
    root,
    num_epochs=20,
    batch_size=4,
    lr=2e-4,
    image_size=256,
    device=None,
    debug=False  # Set default to False to reduce output
):
    # Force GPU usage if available
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Create dataset and dataloader
    train_dataset = OxfordPetDataset(root, image_size=image_size, mode='train', debug=debug)
    val_dataset = OxfordPetDataset(root, image_size=image_size, mode='val', debug=debug)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Dataset sizes - Training: {len(train_dataset)}, Validation: {len(val_dataset)}")
    
    # Instantiate models
    generator = UNetGenerator(in_channels=3, out_channels=1).to(device)
    discriminator = PatchDiscriminator(in_channels=3+1).to(device)
    
    # Optimizers
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr*0.5, betas=(0.5, 0.999))
    
    # Add learning rate schedulers
    g_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(g_optimizer, mode='min', factor=0.5, patience=3)
    d_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(d_optimizer, mode='min', factor=0.5, patience=3)
    
    # For recording metrics
    train_g_losses = []
    train_d_losses = []
    train_adv_losses = []
    train_bce_losses = []
    train_iou_losses = []
    val_accuracies = []
    val_ious = []
    
    # Open a log file for saving losses
    with open('training_log.txt', 'w') as log_file:
        log_file.write("Epoch,G_Loss,Adv_Loss,BCE_Loss,IoU_Loss,D_Loss,Val_Acc,Val_IoU\n")
        
        # Training loop
        for epoch in range(num_epochs):
            print(f"Starting Epoch {epoch+1}/{num_epochs}")
            
            g_loss_running = 0.0
            d_loss_running = 0.0
            adv_loss_running = 0.0
            bce_loss_running = 0.0
            iou_loss_running = 0.0
            
            # Training phase
            generator.train()
            discriminator.train()
            
            for i, (real_img, real_mask) in enumerate(train_loader):
                real_img = real_img.to(device)
                real_mask = real_mask.to(device)
                
                # Skip batches with extremely skewed masks (all 0 or all 1)
                mask_mean = real_mask.mean().item()
                if mask_mean < 0.005 or mask_mean > 0.995:
                    continue
                
                # ----------------------
                #  Train Discriminator
                # ----------------------
                d_optimizer.zero_grad()
                
                # Generate fake masks
                with torch.no_grad():
                    fake_mask = generator(real_img)
                
                # Real pair
                real_input_dis = torch.cat((real_img, real_mask.unsqueeze(1) if real_mask.dim() == 3 else real_mask), dim=1)
                pred_real = discriminator(real_input_dis)
                
                # Fake pair
                fake_input_dis = torch.cat((real_img, fake_mask.detach()), dim=1)
                pred_fake = discriminator(fake_input_dis)
                
                # Discriminator loss
                d_loss = discriminator_loss_fn(pred_real, pred_fake)
                
                # Only update if loss is not too small
                if d_loss.item() > 0.1:
                    d_loss.backward()
                    d_optimizer.step()
                
                # ----------------------
                #  Train Generator
                # ----------------------
                g_optimizer.zero_grad()
                
                # Generate fake masks again
                fake_mask = generator(real_img)
                
                # Ensure mask dimensions are compatible for concatenation
                if real_mask.dim() == 3 and fake_mask.dim() == 4:
                    real_mask_for_loss = real_mask.unsqueeze(1)
                else:
                    real_mask_for_loss = real_mask
                
                # Discriminator's take on fake
                fake_input_dis = torch.cat((real_img, fake_mask), dim=1)
                pred_fake = discriminator(fake_input_dis)
                
                # Generator loss
                g_loss_tuple = generator_loss_fn(
                    pred_fake, fake_mask, real_mask_for_loss, 
                    lambda_adv=1.0, lambda_seg=100, lambda_iou=50
                )
                
                # Unpack loss components
                if isinstance(g_loss_tuple, tuple):
                    g_loss, adv_loss, bce_loss, iou = g_loss_tuple
                else:
                    g_loss = g_loss_tuple
                    adv_loss = torch.tensor(0.0).to(device)
                    bce_loss = torch.tensor(0.0).to(device)
                    iou = torch.tensor(0.0).to(device)
                
                g_loss.backward()
                g_optimizer.step()
                
                # Track losses
                g_loss_running += g_loss.item()
                d_loss_running += d_loss.item()
                adv_loss_running += adv_loss.item()
                bce_loss_running += bce_loss.item()
                iou_loss_running += iou.item()
            
            # Update learning rate schedulers
            if epoch > 0:
                g_scheduler.step(g_loss_running / len(train_loader))
                d_scheduler.step(d_loss_running / len(train_loader))
            
            # Average training losses for this epoch
            epoch_g_loss = g_loss_running / len(train_loader)
            epoch_d_loss = d_loss_running / len(train_loader)
            epoch_adv_loss = adv_loss_running / len(train_loader)
            epoch_bce_loss = bce_loss_running / len(train_loader)
            epoch_iou_loss = iou_loss_running / len(train_loader)
            
            # Store metrics
            train_g_losses.append(epoch_g_loss)
            train_d_losses.append(epoch_d_loss)
            train_adv_losses.append(epoch_adv_loss)
            train_bce_losses.append(epoch_bce_loss)
            train_iou_losses.append(epoch_iou_loss)
            
            # Validation phase
            generator.eval()
            val_accuracy_running = 0.0
            val_iou_running = 0.0
            
            with torch.no_grad():
                for real_img, real_mask in val_loader:
                    real_img = real_img.to(device)
                    real_mask = real_mask.to(device)
                    
                    # Generate fake masks
                    fake_mask = generator(real_img)
                    
                    # Ensure mask dimensions are compatible for metrics calculation
                    if fake_mask.dim() == 4 and fake_mask.size(1) == 1:
                        fake_mask = fake_mask.squeeze(1)
                    
                    # Compute metrics
                    accuracy = compute_accuracy(fake_mask, real_mask)
                    iou = compute_iou(fake_mask, real_mask)
                    
                    val_accuracy_running += accuracy
                    val_iou_running += iou
            
            # Average validation metrics for this epoch
            epoch_val_accuracy = val_accuracy_running / len(val_loader)
            epoch_val_iou = val_iou_running / len(val_loader)
            
            val_accuracies.append(epoch_val_accuracy)
            val_ious.append(epoch_val_iou)
            
            # Visualize predictions every epoch
            visualize_predictions(generator, val_loader, device, epoch, save_dir='./visualizations')
            
            # Logging to console (simplified)
            print(f"Epoch [{epoch+1}/{num_epochs}] - "
                  f"G_Loss: {epoch_g_loss:.4f}, D_Loss: {epoch_d_loss:.4f}, "
                  f"Val_Acc: {epoch_val_accuracy:.4f}, Val_IoU: {epoch_val_iou:.4f}")
            
            # Save to log file
            log_file.write(f"{epoch+1},{epoch_g_loss:.6f},{epoch_adv_loss:.6f},{epoch_bce_loss:.6f},"
                          f"{epoch_iou_loss:.6f},{epoch_d_loss:.6f},{epoch_val_accuracy:.6f},{epoch_val_iou:.6f}\n")
            log_file.flush()
            
            # Save intermediate models every 5 epochs
            if (epoch + 1) % 5 == 0:
                os.makedirs('./saved_models', exist_ok=True)
                torch.save(generator.state_dict(), f'./saved_models/generator_epoch_{epoch+1}.pth')
                torch.save(discriminator.state_dict(), f'./saved_models/discriminator_epoch_{epoch+1}.pth')
                print(f"Saved models at epoch {epoch+1}")
        
    # Plot training losses
    plt.figure(figsize=(15, 12))
    
    # Plot overall losses
    plt.subplot(3, 1, 1)
    plt.plot(range(1, num_epochs+1), train_g_losses, 'b-', label='Generator Loss')
    plt.plot(range(1, num_epochs+1), train_d_losses, 'r-', label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Losses')
    plt.grid(True)
    
    # Plot loss components
    plt.subplot(3, 1, 2)
    plt.plot(range(1, num_epochs+1), train_adv_losses, 'c-', label='Adversarial Loss')
    plt.plot(range(1, num_epochs+1), train_bce_losses, 'y-', label='BCE Loss')
    plt.plot(range(1, num_epochs+1), train_iou_losses, 'k-', label='IoU Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Component')
    plt.legend()
    plt.title('Generator Loss Components')
    plt.grid(True)
    
    # Plot validation metrics
    plt.subplot(3, 1, 3)
    plt.plot(range(1, num_epochs+1), val_accuracies, 'g-', label='Validation Accuracy')
    plt.plot(range(1, num_epochs+1), val_ious, 'm-', label='Validation IoU')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.title('Validation Metrics')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    print("Training metrics plot saved to 'training_metrics.png'")
    plt.close()
    
    # Save metrics to file
    metrics = {
        'generator_loss': train_g_losses,
        'discriminator_loss': train_d_losses,
        'adversarial_loss': train_adv_losses,
        'bce_loss': train_bce_losses,
        'iou_loss': train_iou_losses,
        'validation_accuracy': val_accuracies,
        'validation_iou': val_ious
    }
    
    np.save('training_metrics.npy', metrics)
    print("Training metrics saved to 'training_metrics.npy'")
    
    return generator, discriminator, metrics

if __name__ == "__main__":
    root_dir = "./datasets"
    img_paths = sorted(glob.glob(os.path.join(root_dir, 'images', '*.jpg')))
    mask_paths = sorted(glob.glob(os.path.join(root_dir, 'annotations', 'trimaps', '*.png')))
    
    print("Number of images found:", len(img_paths))
    print("Number of masks found:", len(mask_paths))
    
    # Check dataset structure
    if len(img_paths) == 0 or len(mask_paths) == 0:
        print("ERROR: Could not find images or masks in the dataset directory.")
        print("Please ensure the dataset structure is correct:")
        print("  - datasets/images/*.jpg")
        print("  - datasets/annotations/trimaps/*.png")
        exit(1)
    
    # Force GPU usage if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Train with modified hyperparameters
    generator, discriminator, metrics = train_gan(
        root_dir, 
        num_epochs=20,
        batch_size=4, 
        lr=2e-4,
        device=device,
        debug=False
    )
    
    # Save models
    os.makedirs('./saved_models', exist_ok=True)
    torch.save(generator.state_dict(), './saved_models/generator_final.pth')
    torch.save(discriminator.state_dict(), './saved_models/discriminator_final.pth')
    print("Final models saved to './saved_models/'")