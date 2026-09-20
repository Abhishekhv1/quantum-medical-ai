import math
import torch
import torch.nn as nn

class MedicalCNNBackbone(nn.Module):
    """
    Enhanced feature extractor for Chest Radiographs.
    Uses 3 Conv Blocks + Adaptive Pooling to map (B, 1, 64, 64)
    into a robust 4D latent vector bounded to [-pi, pi] via pi * tanh(x).
    """
    def __init__(self):
        super(MedicalCNNBackbone, self).__init__()
        
        # Block 1: 1 -> 32 channels, 64x64 -> 32x32
        self.b1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Block 2: 32 -> 64 channels, 32x32 -> 16x16
        self.b2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        # Block 3: 64 -> 128 channels, 16x16 -> 8x8
        self.b3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        # Global pooling + non-linear compression to 4D
        self.pool = nn.AdaptiveAvgPool2d((2, 2)) # 128 * 2 * 2 = 512
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(64, 4)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.pool(x)
        raw_4d = self.fc(x)
        return math.pi * torch.tanh(raw_4d)
