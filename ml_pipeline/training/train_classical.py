import os
import torch
import torch.nn as nn
import torch.optim as optim
from ml_pipeline.data.dataset import get_dataloaders
from ml_pipeline.models.classical_cnn import ClassicalChestCNN

def train_classical(epochs=10, batch_size=32, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Classical Model on: {device}")

    train_loader, val_loader, _ = get_dataloaders("data/raw", batch_size=batch_size)
    model = ClassicalChestCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    os.makedirs("ml_pipeline/checkpoints", exist_ok=True)
    checkpoint_path = "ml_pipeline/checkpoints/classical_best.pth"

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
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

        train_acc = correct / total
        train_loss = running_loss / total

        # Validation phase
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
        print(f"Epoch [{epoch+1:02d}/{epochs:02d}] - Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  --> Saved new best checkpoint: {checkpoint_path}")

    print(f"\nClassical Training Complete. Best Validation Accuracy: {best_val_acc*100:.2f}%")

if __name__ == "__main__":
    train_classical(epochs=5, batch_size=32, lr=0.001)
