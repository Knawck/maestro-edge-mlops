import numpy as np
import pandas as pd
import os

# MAESTRO GOLD STANDARD thresholds [cite: 2]
RTT_THRESHOLD    = 35.0    
JITTER_THRESHOLD = 8.0     
LOSS_THRESHOLD   = 3.0     

TOTAL_SAMPLES    = 60000
SEQUENCE_LENGTH  = 20      

def reverting_walk(n_steps, start, target, drift_strength, noise_scale, low_clip, high_clip):
    """Generates a reverting random walk to simulate network drift[cite: 4, 7]."""
    values = np.zeros(n_steps)
    values[0] = start
    for i in range(1, n_steps):
        pull  = drift_strength * (target - values[i - 1])
        noise = np.random.normal(0, noise_scale)
        values[i] = np.clip(values[i - 1] + pull + noise, low_clip, high_clip)
    return values

def generate_dataset(n_samples=TOTAL_SAMPLES, seq_len=SEQUENCE_LENGTH, output_path="data/traces.csv"):
    np.random.seed(42)
    rows = []

    for i in range(n_samples):
        # Generate independent reverting walks for telemetry [cite: 10, 11]
        rtt    = reverting_walk(seq_len, np.random.uniform(5, 80), 20.0, 0.15, 4.0, 1.0, 150.0)
        jitter = reverting_walk(seq_len, np.random.uniform(0, 20), 3.0, 0.15, 2.0, 0.0, 30.0)
        loss   = reverting_walk(seq_len, np.random.uniform(0, 10), 1.0, 0.20, 1.5, 0.0, 20.0)
        ecn    = reverting_walk(seq_len, np.random.uniform(0, 1), 0.2, 0.10, 0.1, 0.0, 1.0)

        # NEW LOGIC: Label 1 = Healthy, Label 0 = Degraded [cite: 80, 83]
        final_rtt, final_jitter, final_loss = rtt[-1], jitter[-1], loss[-1]
        is_healthy = (final_rtt <= RTT_THRESHOLD and 
                      final_jitter <= JITTER_THRESHOLD and 
                      final_loss <= LOSS_THRESHOLD)
        label = 1 if is_healthy else 0

        for t in range(seq_len):
            rows.append({
                "sample_id": i,
                "frame":     t,
                "rtt":       round(rtt[t], 3),
                "jitter":    round(jitter[t], 3),
                "loss":      round(loss[t], 3),
                "ecn":       round(ecn[t], 4),
                "dummy1":    0.0,
                "dummy2":    0.0,
                "label":     label,
            })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset generated → {output_path} (1=Healthy, 0=Degraded)")
    return df

if __name__ == "__main__":
    generate_dataset()