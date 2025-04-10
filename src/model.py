import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import OxfordIIITPet


class WeakSegmentationModel:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Data preparation
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load Oxford-IIIT Pet Dataset
        self.train_dataset = OxfordIIITPet(
            root='data',
            split='train',
            transform=self.transform,
            download=True
        )
        
        self.test_dataset = OxfordIIITPet(
            root='data',
            split='test',
            transform=self.transform,
            download=True
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config['batch_size'],
            shuffle=True
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=config['batch_size'],
            shuffle=False
        )
        
        # Initialize model
        self.model = WeaklySegmentationModel(
            backbone=config['backbone'],
            num_classes=config['num_classes']
        ).to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate']
        )

    def train(self):
        self.model.train()
        for epoch in range(self.config['epochs']):
            for batch_idx, (images, targets) in enumerate(self.train_loader):
                images, targets = images.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                if batch_idx % 100 == 0:
                    print(f'Epoch [{epoch+1}/{self.config["epochs"]}], '
                          f'Step [{batch_idx}/{len(self.train_loader)}], '
                          f'Loss: {loss.item():.4f}')

    def evaluate(self):
        self.model.eval()
        total_metrics = {}
        
        with torch.no_grad():
            for images, targets in self.test_loader:
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = self.model(images)
                metrics = calculate_metrics(outputs, targets)
                
                # Accumulate metrics
                for key, value in metrics.items():
                    total_metrics[key] = total_metrics.get(key, 0) + value
        
        # Average metrics
        avg_metrics = {k: v/len(self.test_loader) for k, v in total_metrics.items()}
        return avg_metrics