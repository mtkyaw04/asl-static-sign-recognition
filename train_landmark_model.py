import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Configuration
DATASET_DIR = "dataset_landmarks"
BATCH_SIZE = 32
EPOCHS = 100
PATIENCE = 10  # For early stopping
LR = 0.0001

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load and preprocess data
def load_data():
    X, y = [], []
    labels = sorted([f.replace('.json', '') for f in os.listdir(DATASET_DIR) if f.endswith('.json')])
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    
    # Check class balance
    class_counts = {}
    
    for label in labels:
        with open(os.path.join(DATASET_DIR, f"{label}.json")) as f:
            samples = json.load(f)
            X.extend(samples)
            y.extend([label_to_index[label]] * len(samples))
            class_counts[label] = len(samples)
    
    print("\nClass distribution:")
    for label, count in class_counts.items():
        print(f"{label}: {count} samples")
    
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    
    return X, y, label_to_index

# Enhanced Model Architecture
class HandSignClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.layers(x)

# Training function with early stopping
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, patience):
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss, train_correct = 0, 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()
        
        # Validation
        model.eval()
        val_loss, val_correct = 0, 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                preds = outputs.argmax(1)
                val_correct += (preds == labels).sum().item()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        train_loss /= len(train_loader)
        train_acc = train_correct / len(train_loader.dataset)
        val_loss /= len(val_loader)
        val_acc = val_correct / len(val_loader.dataset)
        
        print(f"\nEpoch {epoch+1}:")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'model_state': model.state_dict(),
                'label_to_index': label_to_index
            }, "best_landmark_model.pth")
            print("Saved new best model")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Final evaluation
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=label_to_index.keys()))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

# Main execution
if __name__ == "__main__":
    # Load data
    X, y, label_to_index = load_data()
    
    # Split data (80% train, 20% val)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    model = HandSignClassifier(input_size=X.shape[1], num_classes=len(label_to_index))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Train
    print(f"\nTraining on {len(train_dataset)} samples, validating on {len(val_dataset)} samples")
    train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS, PATIENCE)