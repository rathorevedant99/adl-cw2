import matplotlib.pyplot as plt
import re
import numpy as np
import argparse
from pathlib import Path

def extract_loss_from_logs(log_file):
    """Extract loss values from training log file"""
    batch_pattern = r'Batch (\d+)/(\d+), Loss: ([-+]?\d*\.\d+)'
    epoch_pattern = r'Epoch (\d+) completed\. Average loss: ([-+]?\d*\.\d+)'
    
    batch_losses = []
    epoch_losses = []
    
    with open(log_file, 'r') as f:
        for line in f:
            # Extract batch losses
            batch_match = re.search(batch_pattern, line)
            if batch_match:
                batch_idx = int(batch_match.group(1))
                total_batches = int(batch_match.group(2))
                loss = float(batch_match.group(3))
                batch_losses.append((batch_idx, total_batches, loss))
            
            # Extract epoch losses
            epoch_match = re.search(epoch_pattern, line)
            if epoch_match:
                epoch = int(epoch_match.group(1))
                loss = float(epoch_match.group(2))
                epoch_losses.append((epoch, loss))
    
    return batch_losses, epoch_losses

def create_plots(batch_losses, epoch_losses, output_dir):
    """Create and save loss plots"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot epoch losses
    if epoch_losses:
        plt.figure(figsize=(10, 6))
        epochs = [e[0] for e in epoch_losses]
        losses = [e[1] for e in epoch_losses]
        
        plt.plot(epochs, losses, 'b-o', linewidth=2)
        plt.title('Training Loss per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.grid(True)
        
        # Add trendline
        if len(epochs) > 2:
            z = np.polyfit(epochs, losses, min(3, len(epochs)-1))
            p = np.poly1d(z)
            plt.plot(epochs, p(epochs), 'r--', linewidth=1)
        
        plt.tight_layout()
        epoch_plot_path = output_dir / 'epoch_loss.png'
        plt.savefig(epoch_plot_path)
        print(f"Saved epoch loss plot to {epoch_plot_path}")
        plt.close()
    else:
        print("No epoch loss data found in the log file")
    
    # Plot batch losses
    if batch_losses:
        plt.figure(figsize=(12, 6))
        losses = [b[2] for b in batch_losses]
        batches = range(len(losses))
        
        plt.plot(batches, losses, 'g-', alpha=0.7)
        plt.title('Batch Losses During Training')
        plt.xlabel('Batch (by order in log)')
        plt.ylabel('Loss')
        plt.grid(True)
        
        # Add smoothed trendline
        if len(losses) > 10:
            window_size = min(50, len(losses) // 10)
            smoothed = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, window_size-1+len(smoothed)), 
                    smoothed, 'r-', linewidth=2, label=f'Moving average (window={window_size})')
            plt.legend()
        
        plt.tight_layout()
        batch_plot_path = output_dir / 'batch_loss.png'
        plt.savefig(batch_plot_path)
        print(f"Saved batch loss plot to {batch_plot_path}")
        plt.close()
    else:
        print("No batch loss data found in the log file")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate loss plots from training logs")
    parser.add_argument("--log_file", type=str, required=True, help="Path to the training log file")
    parser.add_argument("--output_dir", type=str, default="./plots", help="Directory to save the plots")
    
    args = parser.parse_args()
    
    print(f"Extracting loss values from {args.log_file}...")
    batch_losses, epoch_losses = extract_loss_from_logs(args.log_file)
    
    print(f"Found {len(batch_losses)} batch loss entries and {len(epoch_losses)} epoch loss entries")
    create_plots(batch_losses, epoch_losses, args.output_dir)