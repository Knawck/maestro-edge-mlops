"""
MAESTRO PoC-01 — Audio Latency Measurement
Measures round-trip latency and jitter on localhost loopback.
Run clean first, then run again with Clumsy active to see the difference.
"""

import sounddevice as sd
import numpy as np
import time
import csv
import argparse
import os


SAMPLE_RATE = 48000          # Hz — standard for audio
FRAME_MS    = 2.5            # milliseconds per frame (Opus minimum)
FRAME_SIZE  = int(SAMPLE_RATE * FRAME_MS / 1000)   # = 120 samples


def measure_loopback(duration_sec: int, output_file: str):
    """
    Sends audio frames to loopback and records round-trip time for each frame.
    """
    results = []

    def callback(indata, outdata, frames, time_info, status):
        t_send = time.perf_counter_ns()
        outdata[:] = indata          # echo input straight back out
        t_recv = time.perf_counter_ns()
        rtt_ms = (t_recv - t_send) / 1_000_000
        results.append({
            "frame":  len(results),
            "rtt_ms": round(rtt_ms, 4),
            "ts_ns":  t_send,
        })

    print(f"Measuring for {duration_sec}s → {output_file}")
    print("Press Ctrl+C to stop early.\n")

    with sd.Stream(
        samplerate=SAMPLE_RATE,
        blocksize=FRAME_SIZE,
        channels=1,
        dtype="float32",
        callback=callback,
    ):
        sd.sleep(duration_sec * 1000)

    # Write results to CSV
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "rtt_ms", "ts_ns"])
        writer.writeheader()
        writer.writerows(results)

    # Print quick summary
    rtts = [r["rtt_ms"] for r in results]
    jitter = np.std(rtts)
    print(f"Frames recorded : {len(rtts)}")
    print(f"Mean RTT        : {np.mean(rtts):.3f} ms")
    print(f"p99 RTT         : {np.percentile(rtts, 99):.3f} ms")
    print(f"Jitter (σ)      : {jitter:.3f} ms")
    print(f"Max RTT         : {np.max(rtts):.3f} ms")

    if jitter > 8.0:
        print("\n⚠️  WARNING: Jitter exceeds 8ms — musical timing coherence is broken.")
        print("   This is exactly the problem MAESTRO is designed to solve.\n")
    else:
        print("\n✅  Jitter within acceptable range for real-time audio.\n")

    print(f"Results saved to: {output_file}")
    return rtts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MAESTRO PoC-01: Audio Latency Measurement")
    parser.add_argument("--duration", type=int, default=30,
                        help="Measurement duration in seconds (default: 30)")
    parser.add_argument("--out", type=str, default="results/clean.csv",
                        help="Output CSV file path")
    args = parser.parse_args()

    measure_loopback(args.duration, args.out)