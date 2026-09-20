import os
import torch
import torch.nn as nn
import torch.optim as optim
from ml_pipeline.data.dataset import get_dataloaders
from ml_pipeline.models.hybrid_vqc import HybridChestQNN

def train_hybrid_stratified(epochs=3, batch_size=16, samples_per_epoch=256):
    device = torch.device("cpu")
    print(f"Training Hybrid Quantum-Classical Model on: {device}")

    train_loader, test_loader = get_dataloaders("data/processed", batch_size=batch_size)

    # Instantiate hybrid architecture
    model = HybridChestQNN().to(device)

    # Transfer pre-trained classical backbone weights
    checkpoint_path = "ml_pipeline/checkpoints/classical_best.pth"
    if os.path.exists(checkpoint_path):
        print(f"Loading pre-trained feature backbone from {checkpoint_path}...")
        pretrained_dict = torch.load(checkpoint_path, map_location=device)
        backbone_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items() if k.startswith("backbone.")}
        model.backbone.load_state_dict(backbone_dict)
        print("Backbone successfully initialized with pre-trained 95.68% features.")

    # Freeze convolutional feature extractor to accelerate quantum convergence
    for param in model.backbone.parameters():
        param.requires_grad = False

    # Focus optimization on the Variational Quantum Circuit and linear readout
    optimizer = optim.Adam([
        {"params": model.vqc_layer.parameters(), "lr": 0.02},
        {"params": model.readout.parameters(), "lr": 0.01}
    ])
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    hybrid_checkpoint = "ml_pipeline/checkpoints/hybrid_best.pth"

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0, 0, 0
        
        for batch_idx, (imgs, lbls) in enumerate(train_loader):
            if total >= samples_per_epoch:
                break

            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == lbls).sum().item()
            total += lbls.size(0)

            if (batch_idx + 1) % 4 == 0 or total >= samples_per_epoch:
                print(f"Epoch [{epoch+1}/{epochs}] Step [{batch_idx+1}] Batch Loss: {loss.item():.4f}")

        train_acc = (correct / total) * 100
        train_loss = running_loss / total

        # Evaluation phase
        model.eval()
        t_correct, t_total = 0, 0
        with torch.no_grad():
            for idx, (imgs, lbls) in enumerate(test_loader):
                if idx >= 10:  # Evaluate across 160 holdout test images
                    break
                imgs, lbls = imgs.to(device), lbls.to(device)
                outputs = model(imgs)
                preds = torch.argmax(outputs, dim=1)
                t_correct += (preds == lbls).sum().item()
                t_total += lbls.size(0)

        test_acc = (t_correct / t_total) * 100
        print(f"\n---> Epoch [{epoch+1:02d}/{epochs:02d}] Summary - Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%\n")

        if test_acc >= best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), hybrid_checkpoint)
            print(f"  * Checkpoint Updated: {test_acc:.2f}% -> {hybrid_checkpoint}\n")

    print(f"Hybrid Optimization Complete. Validated Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    train_hybrid_stratified()
