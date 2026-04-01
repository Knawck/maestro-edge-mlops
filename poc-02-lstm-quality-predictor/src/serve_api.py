"""
FastAPI Edge Inference Server
"""
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from model import QualityLSTM

app = FastAPI(title='MAESTRO Quality Predictor')

model = QualityLSTM()
model.load_state_dict(torch.load('models/best.pt', map_location='cpu'))
model.eval() 

class NetMetrics(BaseModel):
    rtt_ms: float
    jitter_ms: float
    loss_pct: float
    ecn_fill: float

@app.post('/predict')
def predict(m: NetMetrics):
    # Normalize the incoming data
    x = torch.tensor([[m.rtt_ms/100, m.jitter_ms/10, m.loss_pct/100, m.ecn_fill, 0.0, 0.0]], dtype=torch.float32)
    x = x.unsqueeze(0).repeat(1, 20, 1)
    
    with torch.no_grad(): 
        score = model(x).item()
        
    return {
        'quality_score': round(score, 3), 
        'alert': score < 0.5, 
        'prediction_horizon_ms': 200
    }