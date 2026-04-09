#!/usr/bin/env python3
"""
Test script for openvla_batched_server.py

Compares batched server (port 5103) against sequential server (port 5101)
to verify identical outputs, then tests concurrent request throughput.

Usage:
    # With both servers running:
    .venvs/sglang/bin/python sims/vla_policies/test_batched_server.py

    # Just test the batched server (skip comparison):
    .venvs/sglang/bin/python sims/vla_policies/test_batched_server.py --batched-only
"""

import argparse
import base64
import json
import time
import requests
import numpy as np
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed


def make_test_image(seed=42):
    rng = np.random.RandomState(seed)
    img = Image.fromarray(rng.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def predict(port, image_b64, instruction, timeout=60):
    payload = {"obs": {"image": image_b64, "instruction": instruction}}
    t0 = time.time()
    resp = requests.post(f"http://localhost:{port}/predict", json=payload, timeout=timeout)
    elapsed = time.time() - t0
    resp.raise_for_status()
    return resp.json(), elapsed


def test_single_match(seq_port, bat_port):
    """Test that single requests produce identical outputs."""
    print("=" * 60)
    print("TEST: Single request — compare outputs")
    print("=" * 60)

    image_b64 = make_test_image(42)
    instructions = ["pick up the red cup", "close the drawer", "move the bowl to the left"]
    all_pass = True

    for instr in instructions:
        r_seq, t_seq = predict(seq_port, image_b64, instr)
        r_bat, t_bat = predict(bat_port, image_b64, instr)
        a_seq = np.array(r_seq["actions"][0])
        a_bat = np.array(r_bat["actions"][0])
        diff = np.max(np.abs(a_seq - a_bat))
        ok = diff < 1e-6
        all_pass = all_pass and ok
        status = "PASS" if ok else f"FAIL (diff={diff:.8f})"
        print(f"  [{status}] '{instr}' (seq={t_seq:.1f}s, bat={t_bat:.1f}s)")
        if not ok:
            for i, (a, b) in enumerate(zip(a_seq, a_bat)):
                if abs(a - b) > 1e-6:
                    print(f"    dim {i}: seq={a:.8f} bat={b:.8f}")

    return all_pass


def test_concurrent(bat_port, n_concurrent=5):
    """Test concurrent requests for batching throughput."""
    print("\n" + "=" * 60)
    print(f"TEST: Concurrent requests ({n_concurrent} parallel)")
    print("=" * 60)

    image_b64 = make_test_image(42)
    instruction = "pick up the red cup"

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n_concurrent) as pool:
        futures = [pool.submit(predict, bat_port, image_b64, instruction) for _ in range(n_concurrent)]
        results = []
        for f in as_completed(futures):
            try:
                result, elapsed = f.result()
                results.append((result, elapsed))
            except Exception as e:
                print(f"  ERROR: {e}")
                results.append(None)
    total = time.time() - t0

    ok_count = sum(1 for r in results if r is not None)
    print(f"  {ok_count}/{n_concurrent} succeeded")
    print(f"  Wall time: {total:.2f}s ({total/n_concurrent:.2f}s/req effective)")

    # Check batch stats
    stats = requests.get(f"http://localhost:{bat_port}/health").json()
    print(f"  Batch stats: {json.dumps(stats['batch_stats'], indent=2)}")

    return ok_count == n_concurrent


def test_sequential_baseline(port, n=5):
    """Sequential baseline for comparison."""
    print(f"\n  Sequential baseline ({n} requests on port {port})...")
    image_b64 = make_test_image(42)
    instruction = "pick up the red cup"
    t0 = time.time()
    for _ in range(n):
        predict(port, image_b64, instruction)
    total = time.time() - t0
    print(f"  Total: {total:.2f}s ({total/n:.2f}s/req)")
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-port", type=int, default=5101, help="Sequential server port")
    parser.add_argument("--bat-port", type=int, default=5103, help="Batched server port")
    parser.add_argument("--batched-only", action="store_true", help="Skip comparison with seq server")
    parser.add_argument("--n-concurrent", type=int, default=5, help="Number of concurrent requests")
    args = parser.parse_args()

    # Check servers are up
    for port, name in [(args.bat_port, "batched")]:
        try:
            h = requests.get(f"http://localhost:{port}/health", timeout=5).json()
            if not h.get("ready"):
                print(f"WARNING: {name} server on {port} not ready: {h}")
                return
        except Exception as e:
            print(f"ERROR: {name} server on {port} not reachable: {e}")
            return

    if not args.batched_only:
        try:
            h = requests.get(f"http://localhost:{args.seq_port}/health", timeout=5).json()
            if not h.get("ready"):
                print(f"WARNING: seq server on {args.seq_port} not ready")
        except Exception:
            print(f"WARNING: seq server on {args.seq_port} not reachable, skipping comparison")
            args.batched_only = True

    # Run tests
    if not args.batched_only:
        match_ok = test_single_match(args.seq_port, args.bat_port)
        if match_ok:
            print("\n  All outputs match!")
        else:
            print("\n  WARNING: Output mismatch detected!")

    concurrent_ok = test_concurrent(args.bat_port, args.n_concurrent)

    if not args.batched_only:
        print("\n" + "=" * 60)
        print("THROUGHPUT COMPARISON")
        print("=" * 60)
        t_seq = test_sequential_baseline(args.seq_port, args.n_concurrent)
        t_bat = test_sequential_baseline(args.bat_port, args.n_concurrent)
        if t_bat > 0:
            print(f"\n  Speedup: {t_seq/t_bat:.2f}x")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
