import torch
import torch.nn as nn
import pennylane as qml
from ml_pipeline.models.backbone import MedicalCNNBackbone
from ml_pipeline.models.quantum_circuit import quantum_net, N_QUBITS, N_LAYERS

class HybridChestQNN(nn.Module):
    """
    Model 2: Hybrid CNN + Variational Quantum Circuit (VQC).
    Shared CNN Backbone -> 4D Bottleneck -> 4-Qubit VQC -> 4 -> 2 Linear Readout.
    """
    def __init__(self):
        super(HybridChestQNN, self).__init__()
        # Shared CNN feature extractor
        self.backbone = MedicalCNNBackbone()

        # Define shape dictionary for variational circuit parameters
        weight_shapes = {"weights": (N_LAYERS, N_QUBITS, 3)}

        # Wrap PennyLane QNode as a PyTorch layer
        self.vqc_layer = qml.qnn.TorchLayer(quantum_net, weight_shapes)

        # Classical readout layer mapping 4 quantum expectation values to 2 classes
        self.readout = nn.Linear(N_QUBITS, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Classical feature extraction: (B, 1, 64, 64) -> (B, 4)
        angles = self.backbone(x)

        # 2. Quantum execution: (B, 4) -> (B, 4)
        q_out = self.vqc_layer(angles)

        # 3. Final classification logits: (B, 4) -> (B, 2)
        logits = self.readout(q_out)
        return logits
