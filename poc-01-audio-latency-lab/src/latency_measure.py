"""
MAESTRO PoC-01 — Audio Latency Measurement (UDP Network Architecture)
Measures round-trip latency and jitter via a local UDP socket loopback.
Run clean first, then run again with Clumsy active to see the difference.
"""

import sounddevice as sd
import numpy as np
import time
import csv
import argparse
import os
import socket
import threading
import queue
import struct

SAMPLE_RATE = 48000          # Hz — standard for audio
FRAME_MS    = 2.5            # milliseconds per frame (Opus minimum)
FRAME_SIZE  = int(SAMPLE_RATE * FRAME_MS / 1000)   # = 120 samples

UDP_IP = "127.0.0.1"
UDP_PORT = 8000

def measure_loopback(duration_sec: int, output_file: str):
    """
    Sends audio frames over a UDP socket and records round-trip time.
    """
    results = []
    play_queue = queue.Queue()
    
    # 1. Initialize UDP Socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    sock.settimeout(0.5) # Allows thread to exit cleanly if no data
    
    running = True

    # 2. Network Listener Thread
    def network_listener():
        frame_count = 0
        while running:
            try:
                # Buffer size 4096 is plenty for our 120-sample payload
                data, _ = sock.recvfrom(4096)
                t_recv = time.perf_counter_ns()
                
                # Unpack the 8-byte timestamp and the remaining audio bytes
                t_send = struct.unpack("Q", data[:8])[0]
                audio_bytes = data[8:]
                
                rtt_ms = (t_recv - t_send) / 1_000_000
                results.append({
                    "frame":  frame_count,
                    "rtt_ms": round(rtt_ms, 4),
                    "ts_ns":  t_send,
                })
                frame_count += 1
                
                # Push the raw audio bytes to the playback queue
                play_queue.put(audio_bytes)
                
            except socket.timeout:
                continue # Loop back around and check 'running' flag
            except Exception as e:
                if running: print(f"Socket error: {e}")

    # Start the listener thread
    listener_thread = threading.Thread(target=network_listener, daemon=True)
    listener_thread.start()

    # 3. Audio Stream Callback
    def audio_callback(indata, outdata, frames, time_info, status):
        t_send = time.perf_counter_ns()
        
        # Pack the timestamp ("Q" = unsigned long long) + the raw audio bytes
        packet = struct.pack("Q", t_send) + indata.tobytes()
        
        # Blast it over the network
        sock.sendto(packet, (UDP_IP, UDP_PORT))
        
        # Try to play whatever the network thread has caught
        try:
            incoming_bytes = play_queue.get_nowait()
            outdata[:] = np.frombuffer(incoming_bytes, dtype=np.float32).reshape(-1, 1)
        except queue.Empty:
            # If the network lagged or dropped a packet, we must output silence
            outdata.fill(0)

    print(f"Measuring for {duration_sec}s → {output_file}")
    print("Press Ctrl+C to stop early.\n")

    try:
        with sd.Stream(
            samplerate=SAMPLE_RATE,
            blocksize=FRAME_SIZE,
            channels=1,
            dtype="float32",
            callback=audio_callback,
        ):
            time.sleep(duration_sec)
    except KeyboardInterrupt:
        print("\nMeasurement stopped by user.")
    finally:
        # 4. Clean up sockets and threads
        running = False
        listener_thread.join()
        sock.close()

    if not results:
        print("Error: No packets were successfully routed back. Check firewall.")
        return []

    # 5. Write results to CSV
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "rtt_ms", "ts_ns"])
        writer.writeheader()
        writer.writerows(results)

    # 6. Print quick summary
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