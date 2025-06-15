import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
from ndlinear import NdLinear
import os
import sys

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_score, model):
        if self.best_score is None:
            self.best_score = val_score
            self.save_checkpoint(model)
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = val_score
            self.save_checkpoint(model)
            self.counter = 0
        return False
    
    def save_checkpoint(self, model):
        """Save model when validation score improves"""
        self.best_weights = model.state_dict().copy()

class NdCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(NdCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 32x32x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16x32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 16x16x64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8x64

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 8x8x128
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 4x4x128
        )
        
        # Use global average pooling to reduce feature maps to 1x1x128
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # 1x1x128
        self.classifier = NdLinear((128,), (num_classes,))  # Replaces nn.Linear(128, 10)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)  # shape: [batch_size, 128, 1, 1]
        x = x.view(x.size(0), -1)  # shape: [batch_size, 128]
        x = self.classifier(x)  # NdLinear: shape -> [batch_size, 10]
        return x


def get_data_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    full_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    val_size = 5000
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, val_loader, test_loader


def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            output = model(inputs)
            preds = torch.argmax(output, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    return correct / total


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, test_loader = get_data_loaders()

    model = NdCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=10, min_delta=0.001, restore_best_weights=True)

    print("Starting training...")
    for epoch in range(1, 100):
        loss = train(model, train_loader, criterion, optimizer, device)
        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: Loss = {loss:.4f}, Val Acc = {val_acc:.4f}")
        
        # Check early stopping
        if early_stopping(val_acc, model):
            print(f"Early stopping triggered at epoch {epoch}")
            print(f"Best validation accuracy: {early_stopping.best_score:.4f}")
            break
    
    # Evaluate on test data after training is complete
    print("\nTraining complete. Evaluating on test data...")
    test_acc = evaluate(model, test_loader, device)
    print(f"Final test accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()