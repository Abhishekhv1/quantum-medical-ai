import pennylane as qml

# 4 qubits corresponding directly to the 4 bottleneck dimensions
N_QUBITS = 4
N_LAYERS = 2

# Initialize the default state-vector simulator device
dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev, interface="torch", diff_method="parameter-shift")
def quantum_net(inputs, weights):
    """
    Parameterized Quantum Circuit (VQC).
    - inputs: shape (4,) corresponding to the CNN bottleneck features in [-pi, pi]
    - weights: shape (n_layers, 4, 3) parameterized single-qubit rotation angles
    """
    # 1. State Preparation: Angle Embedding
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")

    # 2. Variational Ansatz: Repeated rotation + circular CNOT entanglement
    for l in range(weights.shape[0]):
        # Single qubit rotations Rx, Ry, Rz
        for i in range(N_QUBITS):
            qml.Rot(weights[l, i, 0], weights[l, i, 1], weights[l, i, 2], wires=i)

        # Entangling layer (circular CNOTs)
        for i in range(N_QUBITS):
            qml.CNOT(wires=[i, (i + 1) % N_QUBITS])

    # 3. Measurement: Pauli-Z expectation values
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]
