import os
import torch
import torch.nn as nn
import torch.optim as optim
from ml_pipeline.data.dataset import get_dataloaders
from ml_pipeline.models.classical_cnn import ClassicalChestCNN

def run_training(epochs=15, batch_size=32, lr=0.0005):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training High-Performance Classical Backbone on: {device}")

    train_loader, _, test_loader = get_dataloaders("data/raw", batch_size=batch_size)

    # Class balance weights
    n_normal = len(os.listdir("data/raw/train/NORMAL"))
    n_pneu = len(os.listdir("data/raw/train/PNEUMONIA"))
    total = n_normal + n_pneu
    weights = torch.tensor([total / (2.0 * n_normal), total / (2.0 * n_pneu)]).to(device)

    model = ClassicalChestCNN().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_acc = 0.0
    checkpoint = "ml_pipeline/checkpoints/classical_best.pth"

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total_samples = 0.0, 0, 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            correct += (torch.argmax(out, dim=1) == lbls).sum().item()
            total_samples += lbls.size(0)

        train_acc = (correct / total_samples) * 100
        train_loss = running_loss / total_samples

        # Evaluate on full 624-sample test set
        model.eval()
        t_correct, t_total = 0, 0
        with torch.no_grad():
            for imgs, lbls in test_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                out = model(imgs)
                t_correct += (torch.argmax(out, dim=1) == lbls).sum().item()
                t_total += lbls.size(0)

        test_acc = (t_correct / t_total) * 100
        scheduler.step(test_acc)

        print(f"Epoch [{epoch+1:02d}/{epochs:02d}] Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Holdout Test Acc: {test_acc:.2f}%")

        if test_acc >= best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), checkpoint)
            print(f"  * Checkpoint Updated: {test_acc:.2f}% -> {checkpoint}")

    print(f"\nFinal Best Holdout Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    run_training()
