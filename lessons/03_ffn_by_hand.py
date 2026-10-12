"""Lesson 3 - A layer is matrix multiplication.

Walks one token's embedding through block 0's feed-forward network with the real weights,
printing every intermediate the lesson quotes: RMS, a matvec checked against a plain loop,
SiLU gating statistics, the residual add and its cosine, the FLOP count, and matvec vs
matmul timing on whatever CPU runs it.

    uv run python lessons/03_ffn_by_hand.py MODEL.gguf

Note: in the real model this FFN receives the output of block 0's attention half. Feeding it
the raw embedding keeps the demo to one half of one block.
"""

import argparse
import time

import numpy as np

from llmbytes import GGUFModel, cosine, rms_norm, silu


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--token", default=" cat")
    ap.add_argument("--batch", type=int, default=256, help="tokens for the matmul timing")
    args = ap.parse_args()

    m = GGUFModel(args.model)
    eps = m.dim("attention.layer_norm_rms_epsilon")
    d, f = m.dim("embedding_length"), m.dim("feed_forward_length")
    print(f"embedding_length={d} feed_forward_length={f} block_count={m.dim('block_count')} rms eps={eps}")

    ids = m.encode(args.token)
    assert len(ids) == 1, f"{args.token!r} is not a single token: {ids}"
    x = m.dequant_row("token_embd.weight", ids[0])

    Wg, Wu, Wd = (m.dequant(f"blk.0.ffn_{n}.weight") for n in ("gate", "up", "down"))
    gain = m.dequant("blk.0.ffn_norm.weight")
    print("shapes (rows x cols): gate", Wg.shape, "up", Wu.shape, "down", Wd.shape)

    print("\n== 1. RMSNorm ==")
    print("  first 3 elements:", np.round(x[:3], 4).tolist(), " squares:", np.round(x[:3] ** 2, 4).tolist())
    ms = float(np.mean(x * x))
    rms = float(np.sqrt(ms + eps))
    print(f"  mean of squares = {ms:.5f}   RMS = {rms:.4f}")
    print(f"  RMS after dividing by it = {float(np.sqrt(np.mean((x / rms) ** 2))):.4f}")
    print(f"  length = RMS x sqrt({d}) = {rms:.4f} x {np.sqrt(d):.0f} = {rms * np.sqrt(d):.2f}"
          f"  (direct: {float(np.linalg.norm(x)):.2f})")
    xn = rms_norm(x, gain, eps)

    print("\n== 2. two matvecs up ==")
    g, u = Wg @ xn, Wu @ xn
    by_hand = 0.0
    for c in range(d):
        by_hand += float(Wg[0, c]) * float(xn[c])
    print(f"  g[0] via NumPy = {float(g[0]):.6f}   via a plain loop = {by_hand:.6f}")

    print("\n== 3. SiLU gating ==")
    for z in (-4, -2, -1, 0, 1, 2, 4):
        print(f"  SiLU({z:+d}) = {float(silu(np.float64(z))):+.3f}")
    s = silu(g)
    print(f"  gates negative: {np.mean(g < 0):.0%}   |SiLU|<0.1: {np.mean(np.abs(s) < 0.1):.0%}"
          f"   |SiLU|<0.01: {np.mean(np.abs(s) < 0.01):.0%}   most negative SiLU: {float(s.min()):.3f}")
    h = s * u

    print("\n== 4-5. down, then the residual ==")
    out = Wd @ h
    y = x + out
    print(f"  length x {np.linalg.norm(x):.2f}, adjustment {np.linalg.norm(out):.2f}, y {np.linalg.norm(y):.2f}")
    print(f"  dot(x, y) = {float(x @ y):.2f}; lengths multiplied = {float(np.linalg.norm(x) * np.linalg.norm(y)):.2f}")
    c = cosine(x, y)
    print(f"  cosine = {c:.4f}  (about {np.degrees(np.arccos(c)):.0f} degrees)")

    print("\n== FLOPs for one token through one whole block ==")
    mat = 2 * (4 * d * d + 3 * d * f)
    elem = 2 * 5 * d + 6 * f
    print(f"  matvecs {mat:,}   everything else ~{elem:,}   matvec share {mat / (mat + elem):.2%}")

    print(f"\n== matvec vs matmul on this CPU ({Wg.nbytes / 1e6:.0f} MB FP32 matrix) ==")
    for _ in range(3):
        Wg @ xn
    n = 20
    t0 = time.perf_counter()
    for _ in range(n):
        Wg @ xn
    t1 = (time.perf_counter() - t0) / n
    X = np.random.default_rng(0).standard_normal((d, args.batch)).astype(np.float32)
    for _ in range(2):
        Wg @ X
    n = 5
    t0 = time.perf_counter()
    for _ in range(n):
        Wg @ X
    tb = (time.perf_counter() - t0) / n
    flops = 2 * Wg.shape[0] * Wg.shape[1]
    print(f"  1 token:    {t1 * 1e3:6.2f} ms  {Wg.nbytes / t1 / 1e9:6.1f} GB/s  {flops / t1 / 1e9:6.1f} GFLOP/s")
    print(f"  {args.batch} tokens: {tb * 1e3:6.2f} ms  {flops * args.batch / tb / 1e9:6.1f} GFLOP/s"
          f"  ({tb / t1:.1f}x the time for {args.batch}x the work)")


if __name__ == "__main__":
    main()
