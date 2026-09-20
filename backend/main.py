import io
import time
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms

from ml_pipeline.models.classical_cnn import ClassicalChestCNN
from ml_pipeline.models.hybrid_vqc import HybridChestQNN

app = FastAPI(title="Hybrid Quantum-Classical Medical AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cpu")
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

classical_model = ClassicalChestCNN().to(device)
hybrid_model = HybridChestQNN().to(device)

@app.on_event("startup")
def load_checkpoints():
    classical_ckpt = "ml_pipeline/checkpoints/classical_best.pth"
    hybrid_ckpt = "ml_pipeline/checkpoints/hybrid_best.pth"

    try:
        classical_model.load_state_dict(torch.load(classical_ckpt, map_location=device))
        classical_model.eval()
        print(f"[OK] Classical baseline loaded: {classical_ckpt}")
    except Exception as e:
        print(f"[WARN] Failed to load classical model: {e}")

    try:
        hybrid_model.load_state_dict(torch.load(hybrid_ckpt, map_location=device))
        hybrid_model.eval()
        print(f"[OK] Hybrid Quantum model loaded: {hybrid_ckpt}")
    except Exception as e:
        print(f"[WARN] Failed to load hybrid model: {e}")

@app.get("/api/health")
def health():
    return {"status": "active", "service": "Quantum-Classical Medical AI Engine"}

@app.post("/api/predict")
async def predict_chest_xray(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        tensor_img = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            # Classical inference
            t0 = time.perf_counter()
            c_logits = classical_model(tensor_img)
            c_probs = torch.softmax(c_logits, dim=1)[0]
            c_latency = round((time.perf_counter() - t0) * 1000, 2)
            c_pred = torch.argmax(c_probs).item()

            # Hybrid Quantum inference
            t1 = time.perf_counter()
            bottleneck_angles = hybrid_model.backbone(tensor_img)
            quantum_expvals = hybrid_model.vqc_layer(bottleneck_angles)
            q_logits = hybrid_model.readout(quantum_expvals)
            q_probs = torch.softmax(q_logits, dim=1)[0]
            q_latency = round((time.perf_counter() - t1) * 1000, 2)
            q_pred = torch.argmax(q_probs).item()

        return {
            "filename": file.filename,
            "classical": {
                "diagnosis": CLASS_NAMES[c_pred],
                "confidence": round(float(c_probs[c_pred].item()) * 100, 2),
                "probabilities": {
                    "NORMAL": round(float(c_probs[0].item()) * 100, 2),
                    "PNEUMONIA": round(float(c_probs[1].item()) * 100, 2)
                },
                "latency_ms": c_latency
            },
            "quantum": {
                "diagnosis": CLASS_NAMES[q_pred],
                "confidence": round(float(q_probs[q_pred].item()) * 100, 2),
                "probabilities": {
                    "NORMAL": round(float(q_probs[0].item()) * 100, 2),
                    "PNEUMONIA": round(float(q_probs[1].item()) * 100, 2)
                },
                "latency_ms": q_latency,
                "telemetry": {
                    "qubits": 4,
                    "ansatz_layers": 2,
                    "angles_rad": [round(float(a), 4) for a in bottleneck_angles[0].tolist()],
                    "pauli_z_expectations": [round(float(z), 4) for z in quantum_expvals[0].tolist()]
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
