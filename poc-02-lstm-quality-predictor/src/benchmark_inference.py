"""
MAESTRO PoC-02 — Inference Latency Benchmark
Verifies that p99 inference time is under 10ms on CPU.
This is a hard requirement for MAESTRO Layer 2.
"""

import torch
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from model import load_model

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best.pt")


def benchmark(n_iterations: int = 1000):
    print(f"Loading model from {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    model.eval()

    dummy_input = torch.randn(1, 20, 6)
    latencies   = []

    print(f"Running {n_iterations} inference calls ...\n")

    # Warm-up — first few calls are slower due to PyTorch JIT compilation
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)

    # Actual benchmark
    for i in range(n_iterations):
        t0 = time.perf_counter_ns()
        with torch.no_grad():
            _ = model(dummy_input)
        latencies.append((time.perf_counter_ns() - t0) / 1_000_000)

    latencies.sort()
    p50  = latencies[int(n_iterations * 0.50)]
    p95  = latencies[int(n_iterations * 0.95)]
    p99  = latencies[int(n_iterations * 0.99)]
    mean = np.mean(latencies)

    print(f"Results over {n_iterations} iterations:")
    print(f"  Mean   : {mean:.3f} ms")
    print(f"  p50    : {p50:.3f} ms")
    print(f"  p95    : {p95:.3f} ms")
    print(f"  p99    : {p99:.3f} ms")
    print(f"  Max    : {max(latencies):.3f} ms")

    target_ms = 10.0
    if p99 < target_ms:
        print(f"\n✅  PASS — p99 ({p99:.2f}ms) is under {target_ms}ms target")
    else:
        print(f"\n❌  FAIL — p99 ({p99:.2f}ms) exceeds {target_ms}ms target")

    return p99


if __name__ == "__main__":
    benchmark(1000)