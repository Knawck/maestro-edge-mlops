from fastapi import FastAPI
from pydantic import BaseModel
import torch
from model import QualityLSTM

app = FastAPI()
model = QualityLSTM()
model.load_state_dict(torch.load('models/best.pt', map_location='cpu'))
model.eval()

class Metrics(BaseModel):
    rtt: float
    jitter: float
    loss: float
    ecn: float

@app.post('/predict')
def predict(m: Metrics):
    # Prepare sequence of 20 identical frames for a single point prediction
    x = torch.tensor([[m.rtt/100, m.jitter/10, m.loss/100, m.ecn, 0, 0]], dtype=torch.float32)
    x = x.unsqueeze(0).repeat(1, 20, 1)
    
    with torch.no_grad():
        score = model(x).item()
        
    return {"quality": round(score, 3), "alert": score < 0.5}