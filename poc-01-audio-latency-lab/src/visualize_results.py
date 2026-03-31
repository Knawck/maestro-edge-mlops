"""
MAESTRO PoC-01 — Results Visualiser
Plots baseline vs degraded RTT and jitter side by side.
Run this after you have both results/clean.csv and results/degraded.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import argparse
import os


JITTER_THRESHOLD_MS = 8.0    # above this, musical timing breaks down


def load(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}\nRun latency_measure.py first.")
    df = pd.read_csv(path)
    df["jitter_rolling"] = df["rtt_ms"].rolling(window=20).std().fillna(0)
    return df


def plot(baseline_path: str, degraded_path: str, out_path: str = "results/comparison.png"):
    baseline = load(baseline_path)
    degraded = load(degraded_path)

    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(
        "MAESTRO PoC-01 — Audio Latency: Baseline vs Degraded Network\n"
        "Demonstrates why jitter > 8ms destroys real-time musical collaboration",
        fontsize=13, fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

    colors = {"baseline": "#1D4ED8", "degraded": "#DC2626"}

    # ── Top left: RTT over time ───────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(baseline["frame"], baseline["rtt_ms"],
             color=colors["baseline"], linewidth=0.6, alpha=0.8, label="Baseline")
    ax1.plot(degraded["frame"], degraded["rtt_ms"],
             color=colors["degraded"], linewidth=0.6, alpha=0.8, label="Degraded")
    ax1.axhline(20, color="orange", linestyle="--", linewidth=1, label="20ms target")
    ax1.set_title("RTT Over Time", fontweight="bold")
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("RTT (ms)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ── Top right: RTT distribution histogram ────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(baseline["rtt_ms"], bins=50, color=colors["baseline"],
             alpha=0.7, label="Baseline", density=True)
    ax2.hist(degraded["rtt_ms"], bins=50, color=colors["degraded"],
             alpha=0.7, label="Degraded", density=True)
    ax2.set_title("RTT Distribution", fontweight="bold")
    ax2.set_xlabel("RTT (ms)")
    ax2.set_ylabel("Density")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ── Bottom left: rolling jitter ───────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(baseline["frame"], baseline["jitter_rolling"],
             color=colors["baseline"], linewidth=0.8, label="Baseline jitter")
    ax3.plot(degraded["frame"], degraded["jitter_rolling"],
             color=colors["degraded"], linewidth=0.8, label="Degraded jitter")
    ax3.axhline(JITTER_THRESHOLD_MS, color="red", linestyle="--", linewidth=1.5,
                label=f"{JITTER_THRESHOLD_MS}ms collapse threshold")
    ax3.fill_between(degraded["frame"], JITTER_THRESHOLD_MS,
                     degraded["jitter_rolling"].clip(lower=JITTER_THRESHOLD_MS),
                     color="red", alpha=0.15, label="Above threshold")
    ax3.set_title("Rolling Jitter σ  (window=20 frames)", fontweight="bold")
    ax3.set_xlabel("Frame")
    ax3.set_ylabel("Jitter σ (ms)")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ── Bottom right: summary stats table ────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")

    def stats(df, label):
        rtts = df["rtt_ms"]
        return [
            label,
            f"{rtts.mean():.2f} ms",
            f"{np.percentile(rtts, 99):.2f} ms",
            f"{rtts.std():.2f} ms",
            f"{rtts.max():.2f} ms",
            "✅ OK" if rtts.std() < JITTER_THRESHOLD_MS else "❌ BROKEN",
        ]

    rows = [
        stats(baseline, "Baseline"),
        stats(degraded,  "Degraded"),
    ]
    col_labels = ["Scenario", "Mean RTT", "p99 RTT", "Jitter σ", "Max RTT", "Music OK?"]
    table = ax4.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 2)

    # colour the header row
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#111827")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # colour the Music OK? column
    # colour the Music OK? column
    last_col_idx = len(col_labels) - 1  # Get the index of the last column
    for i, row in enumerate(rows, 1):
        color = "#D1FAE5" if "✅" in row[-1] else "#FEE2E2"
        # Use the explicit index instead of -1
        table[i, last_col_idx].set_facecolor(color)

    ax4.set_title("Summary Statistics", fontweight="bold", pad=20)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {out_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MAESTRO PoC-01: Visualise Results")
    parser.add_argument("--baseline", default="results/clean.csv")
    parser.add_argument("--degraded", default="results/degraded.csv")
    parser.add_argument("--out",      default="results/comparison.png")
    args = parser.parse_args()

    plot(args.baseline, args.degraded, args.out)