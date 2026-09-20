import torch
import torch.nn as nn
from ml_pipeline.models.backbone import MedicalCNNBackbone

class ClassicalChestCNN(nn.Module):
    """
    Model 1: Full classical baseline architecture.
    Composed of the shared CNN backbone + a standard 4 -> 2 linear head.
    """
    def __init__(self):
        super(ClassicalChestCNN, self).__init__()
        self.backbone = MedicalCNNBackbone()
        self.classifier = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract 4D bottleneck features
        features = self.backbone(x)
        # Produce class logits: [Batch_Size, 2]
        logits = self.classifier(features)
        return logits
