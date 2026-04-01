import pandas as pd
import numpy as np
import argparse
import os

def generate_data(samples: int, out_path: str):
    print(f"Generating {samples} network traces...")
    np.random.seed(42)
    
    # Simulate normal network traffic
    rtt_ms = np.random.normal(loc=15, scale=5, size=samples) + np.random.exponential(scale=10, size=samples)
    jitter_ms = np.random.exponential(scale=3, size=samples)
    loss_pct = np.random.poisson(lam=1, size=samples)
    ecn_fill = np.random.uniform(0, 1, size=samples)
    
    # Inject artificial network crashes
    bad_events = np.random.choice([True, False], size=samples, p=[0.1, 0.9])
    rtt_ms[bad_events] += np.random.normal(50, 20, size=sum(bad_events))
    jitter_ms[bad_events] += np.random.normal(15, 5, size=sum(bad_events))
    loss_pct[bad_events] += np.random.randint(5, 20, size=sum(bad_events))
    
    # The Target: 1 (Good Quality) if healthy, 0 (Bad Quality) if broken
    quality_label = ((rtt_ms < 30) & (jitter_ms < 8.0) & (loss_pct < 5)).astype(int)
    
    df = pd.DataFrame({
        "rtt_ms": np.clip(rtt_ms, 0, 200),
        "jitter_ms": np.clip(jitter_ms, 0, 50),
        "loss_pct": np.clip(loss_pct, 0, 100),
        "ecn_fill": ecn_fill,
        "dummy_1": 0.0, 
        "dummy_2": 0.0,
        "label": quality_label
    })
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Data saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--out", type=str, default="data/traces.csv")
    args = parser.parse_args()
    generate_data(args.samples, args.out)