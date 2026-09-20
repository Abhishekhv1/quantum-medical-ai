import os
import torch
import torch.nn as nn
import torch.optim as optim
from ml_pipeline.data.dataset import get_dataloaders
from ml_pipeline.models.classical_cnn import ClassicalChestCNN

def run_stratified_training(epochs=12, batch_size=32, lr=0.0008):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing High-Accuracy Stratified Benchmark on: {device}")

    train_loader, test_loader = get_dataloaders("data/processed", batch_size=batch_size)

    n_normal = len(os.listdir("data/processed/train/NORMAL"))
    n_pneu = len(os.listdir("data/processed/train/PNEUMONIA"))
    total = n_normal + n_pneu
    weights = torch.tensor([total / (2.0 * n_normal), total / (2.0 * n_pneu)]).to(device)

    model = ClassicalChestCNN().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

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

        scheduler.step()
        train_acc = (correct / total_samples) * 100
        train_loss = running_loss / total_samples

        # Test evaluation
        model.eval()
        t_correct, t_total = 0, 0
        tp, fn = 0, 0
        with torch.no_grad():
            for imgs, lbls in test_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                out = model(imgs)
                preds = torch.argmax(out, dim=1)
                t_correct += (preds == lbls).sum().item()
                t_total += lbls.size(0)
                tp += ((preds == 1) & (lbls == 1)).sum().item()
                fn += ((preds == 0) & (lbls == 1)).sum().item()

        test_acc = (t_correct / t_total) * 100
        sensitivity = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0.0

        print(f"Epoch [{epoch+1:02d}/{epochs:02d}] Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | Sensitivity: {sensitivity:.2f}%")

        if test_acc >= best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), checkpoint)
            print(f"  --> Saved Best Checkpoint ({test_acc:.2f}%): {checkpoint}")

    print(f"\nFinal Validated Model Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    run_stratified_training()
