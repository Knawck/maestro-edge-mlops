import pandas as pd
import numpy as np
import os

def generate_stateful_data(samples=60000, out_path="data/traces.csv"):
    print("Generating Stateful Network Patterns for Maestro...")
    np.random.seed(42)
    
    # Start with "Good" network values
    rtt, jitter, loss = [15.0], [2.0], [0.5]
    
    for _ in range(1, samples):
        # Create a "Trend" (Random Walk) - each point depends on the last
        rtt.append(max(5, rtt[-1] + np.random.normal(0, 2)))
        jitter.append(max(0.1, jitter[-1] + np.random.normal(0, 0.5)))
        loss.append(max(0, loss[-1] + np.random.normal(0, 0.2)))

    df = pd.DataFrame({
        "rtt": rtt, "jitter": jitter, "loss": loss,
        "ecn": np.random.uniform(0, 1, samples),
        "d1": 0.0, "d2": 0.0
    })
    
    # Harder Logic: Must pass all 3 tests to be "Good" (1)
    df['label'] = ((df['rtt'] < 35) & (df['jitter'] < 8.0) & (df['loss'] < 3)).astype(int)
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Stateful data saved to {out_path}")

if __name__ == "__main__":
    generate_stateful_data()