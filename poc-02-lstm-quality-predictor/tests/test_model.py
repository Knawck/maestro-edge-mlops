"""
MAESTRO PoC-02 — Model Tests
Run with: .venv\Scripts\pytest.exe tests/ -v
"""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import get_model, load_model #type:ignore

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best.pt")


def test_output_shape():
    model = get_model()
    x = torch.randn(4, 20, 6)
    out = model(x)
    assert out.shape == (4, 1), f"Expected (4,1), got {out.shape}"


def test_output_range():
    model = get_model()
    x = torch.randn(8, 20, 6)
    out = model(x)
    assert out.min().item() >= 0.0, "Score below 0.0"
    assert out.max().item() <= 1.0, "Score above 1.0"


def test_inference_latency():
    """p99 inference must be under 10ms — MAESTRO Layer 2 hard requirement."""
    import time
    model = get_model()
    model.eval()
    x = torch.randn(1, 20, 6)

    latencies = []
    for _ in range(200):
        t0 = time.perf_counter_ns()
        with torch.no_grad():
            _ = model(x)
        latencies.append((time.perf_counter_ns() - t0) / 1_000_000)

    p99 = sorted(latencies)[197]
    assert p99 < 10.0, f"p99 latency {p99:.2f}ms exceeds 10ms target"


def test_healthy_input_gives_high_score():
    """Healthy network metrics should produce score > 0.5."""
    if not os.path.exists(MODEL_PATH):
        pytest.skip("Trained model not found — run train_lstm.py first")
    model = load_model(MODEL_PATH)
    # Normalised healthy values: RTT=10/150, Jitter=1/30, Loss=0.1/20
    healthy = torch.tensor([[[10/150, 1/30, 0.1/20, 0.1, 0, 0]] * 20])
    with torch.no_grad():
        score = model(healthy).item()
    assert score > 0.5, f"Expected healthy score > 0.5, got {score:.4f}"


def test_degraded_input_gives_low_score():
    """Degraded network metrics should produce score < 0.5."""
    if not os.path.exists(MODEL_PATH):
        pytest.skip("Trained model not found — run train_lstm.py first")
    model = load_model(MODEL_PATH)
    # Normalised degraded values: RTT=100/150, Jitter=20/30, Loss=15/20
    degraded = torch.tensor([[[100/150, 20/30, 15/20, 0.9, 0, 0]] * 20])
    with torch.no_grad():
        score = model(degraded).item()
    assert score < 0.5, f"Expected degraded score < 0.5, got {score:.4f}"