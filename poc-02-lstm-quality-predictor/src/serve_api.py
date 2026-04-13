import time
import os
import sys
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add src to path for model import [cite: 50]
sys.path.insert(0, os.path.dirname(__file__))
from model import load_model

RTT_NORM, JITTER_NORM, LOSS_NORM, ECN_NORM = 150.0, 30.0, 20.0, 1.0
SEQ_LEN = 20
ALERT_THRESHOLD = 0.5 

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best.pt")

try:
    model = load_model(MODEL_PATH) # [cite: 51]
except:
    model = None

app = FastAPI(title="MAESTRO Quality Predictor")

class NetworkMetrics(BaseModel):
    rtt_ms: float; jitter_ms: float; loss_pct: float; ecn_fill: float # [cite: 53]

@app.post("/predict")
def predict(m: NetworkMetrics):
    if not model: raise HTTPException(status_code=503, detail="Model missing")
    
    t_start = time.perf_counter_ns()
    # Normalize features for LSTM ingestion [cite: 57, 93]
    frame = np.array([m.rtt_ms/RTT_NORM, m.jitter_ms/JITTER_NORM, m.loss_pct/LOSS_NORM, m.ecn_fill/ECN_NORM, 0, 0], dtype=np.float32)
    x = torch.tensor(np.tile(frame, (SEQ_LEN, 1))[np.newaxis, :, :], dtype=torch.float32)
    
    with torch.no_grad():
        score = model(x).item()
    
    # Logic: Score < 0.5 means quality has dropped below the threshold [cite: 58, 92]
    alert = score < ALERT_THRESHOLD 
    
    return {
        "quality_score": round(score, 4),
        "alert": alert,
        "status": "Healthy" if not alert else "Degraded",
        "interpretation": "Network healthy" if not alert else "Network degraded - trigger fallback",
        "inference_ms": round((time.perf_counter_ns() - t_start) / 1_000_000, 3) # [cite: 58]
    }