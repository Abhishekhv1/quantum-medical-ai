import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset
from ml_pipeline.data.dataset import get_dataloaders
from ml_pipeline.models.hybrid_vqc import HybridChestQNN

def train_hybrid(epochs=3, batch_size=16, lr_classical=0.001, lr_quantum=0.01, samples_per_epoch=256):
    device = torch.device("cpu") # Quantum simulation is optimized on CPU state-vector
    print(f"Training Hybrid Model on: {device}")

    train_loader, val_loader, _ = get_dataloaders("data/raw", batch_size=batch_size)
    
    # Use a representative subset per epoch to balance simulation overhead
    train_indices = list(range(min(samples_per_epoch, len(train_loader.dataset))))
    subset_train = Subset(train_loader.dataset, train_indices)
    subset_loader = torch.utils.data.DataLoader(subset_train, batch_size=batch_size, shuffle=True)

    model = HybridChestQNN().to(device)
    criterion = nn.CrossEntropyLoss()

    # Differential parameter optimization
    optimizer = optim.Adam([
        {"params": model.backbone.parameters(), "lr": lr_classical},
        {"params": model.vqc_layer.parameters(), "lr": lr_quantum},
        {"params": model.readout.parameters(), "lr": lr_classical}
    ])

    best_val_acc = 0.0
    os.makedirs("ml_pipeline/checkpoints", exist_ok=True)
    checkpoint_path = "ml_pipeline/checkpoints/hybrid_best.pth"

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for batch_idx, (images, labels) in enumerate(subset_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            print(f"Epoch [{epoch+1}/{epochs}] Step [{batch_idx+1}/{len(subset_loader)}] Batch Loss: {loss.item():.4f}")

        train_acc = correct / total
        train_loss = running_loss / total

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total if val_total > 0 else 0.0
        print(f"\n===> Epoch [{epoch+1:02d}/{epochs:02d}] Summary - Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%\n")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  --> Saved new best hybrid checkpoint: {checkpoint_path}\n")

    print(f"Hybrid Training Complete. Best Validation Accuracy: {best_val_acc*100:.2f}%")

if __name__ == "__main__":
    train_hybrid(epochs=3, batch_size=16)
