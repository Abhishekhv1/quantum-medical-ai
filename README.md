# Hybrid Quantum-Classical Medical AI

An end-to-end medical image diagnostic platform that integrates a PyTorch Convolutional Neural Network (CNN) backbone with a 4-qubit Parameterized Variational Quantum Circuit (VQC) implemented in PennyLane. Built for parameter-efficient chest radiograph classification (Normal vs. Pneumonia) with interactive dual-model inference and real-time quantum telemetry.

---

## Executive Summary
- **Domain:** Quantum Machine Learning (QML) & Computer Vision in Healthcare.
- **Problem Addressed:** High parameter footprints and overconfident misclassifications in traditional medical deep learning models deployed on compute-constrained edge devices.
- **Core Architecture:** Shared 3-stage PyTorch CNN feature extractor -> 4D bottleneck bounded to [-pi, pi] via pi * tanh -> 4-qubit Variational Quantum Circuit with circular CNOT entanglement -> Linear Readout.
- **Inference Engine:** Asynchronous FastAPI backend coupled with a Vite + React + Tailwind CSS diagnostic dashboard.

---

## System Architecture

```text
                                  [ Input Image: 64x64 Grayscale ]
                                                 |
                                                 v
                             [ PyTorch CNN Feature Extractor (Backbone) ]
                             (3x Conv2D + BatchNorm + ReLU + MaxPool2D)
                                                 |
                                                 v
                                   [ Adaptive Average Pooling ]
                                                 |
                                                 v
                                     [ 4D Bottleneck Vector ]
                                                 |
                                     [ Angle Scaling: pi * tanh ]
                                                 |
                                +----------------+----------------+
                                |                                 |
                                v                                 v
                     [ Path A: Classical Head ]       [ Path B: Quantum Layer ]
                       (Linear 4 -> 2 Logits)           (PennyLane 4-Qubit VQC)
                                |                                 |
                                |                      Angle Embedding: Ry(phi_i)
                                |                                 |
                                |                      2x Variational Layers: Rot
                                |                                 |
                                |                      Ring Entanglement: CNOT
                                |                                 |
                                |                      Expectation: <Z_i> in [-1, 1]
                                |                                 |
                                |                      Readout Head: Linear 4 -> 2
                                |                                 |
                                +----------------+----------------+
                                                 |
                                                 v
                                    [ Softmax Classification ]
                                     (NORMAL vs PNEUMONIA)

---

## Technology Stack

- **Quantum AI & ML:** PennyLane (VQC Design, TorchLayer integration & Parameter-Shift Rule), PyTorch & Torchvision (CNN Backbone & Layer Architecture), Qiskit, default.qubit simulator.
- **Backend & Serving:** Python 3.12, FastAPI (REST API), Uvicorn (ASGI Server), Pillow (PIL Image Processing).
- **Frontend & UI:** React 19, Vite, Tailwind CSS v4, Axios, Lucide React Icons.
- **Dataset & Tools:** Kaggle Pediatric Chest X-Ray Dataset (Stratified 85/15 Split), NumPy, PowerShell.

---

## Getting Started

### Prerequisites
- **Python 3.12+**
- **Node.js (v18+) & npm**
- **Git**

### Installation

1. **Clone the Repository:**
   git clone https://github.com/YOUR_USERNAME/hybrid-quantum-medical-ai.git
   cd hybrid-quantum-medical-ai

2. **Set Up Python Virtual Environment:**
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install --upgrade pip
   pip install torch torchvision pennylane qiskit fastapi uvicorn pillow axios

3. **Set Up Frontend Dependencies:**
   cd frontend
   npm install
   cd ..

---

## Running the Application

Launch the backend and frontend services across two separate terminal instances:

### Terminal 1: Backend API Service
From the repository root folder:
uvicorn backend.main:app --reload --port 8000
- API Health Endpoint: http://127.0.0.1:8000/api/health
- Swagger Documentation: http://127.0.0.1:8000/docs

### Terminal 2: Diagnostic Dashboard
From the frontend directory:
cd frontend
npm run dev
- Open browser at http://localhost:5173/

---

## Telemetry & Features

- **Dual-Inference Comparison:** Side-by-side diagnostic classification, confidence percentage, and latency benchmarking (ms) comparing the classical baseline vs. the hybrid quantum circuit.
- **Quantum State Telemetry:** Live numerical output of:
  - Latent rotation angles phi_i in [-pi, pi] fed to each qubit.
  - Pauli-Z observable expectation values <Z_i> in [-1, 1].
- **Conservative Calibration:** Demonstrates bounded quantum logit behavior, resisting overconfident misclassifications typical of deep unregularized classical networks.

---
