"""Lesson 6 - Attention and the KV cache.

Rebuilds the whole model in NumPy from the GGUF weights - embedding, 32 blocks of attention and
feed-forward, final norm, output matrix, softmax - and uses the rebuild to:

  1. check its next-token prediction against llama.cpp's (pass --server to compare directly),
  2. capture every attention weight, for every layer and head,
  3. test the KV cache: process one more token against the cache, versus recomputing everything.

    uv run python lessons/06_attention_rebuild.py MODEL.gguf [--server http://127.0.0.1:8099]

Runs on the CPU. Most of its time is dequantizing ~7 GB of weights on every pass.
"""

import argparse
import math
import time

import numpy as np

from llmbytes import GGUFModel, rms_norm, silu, softmax
from llmbytes.server import LlamaServer


class Rebuild:
    def __init__(self, m: GGUFModel):
        self.m = m
        self.L = m.dim("block_count")
        self.D = m.dim("embedding_length")
        self.H = m.dim("attention.head_count")
        self.HKV = m.dim("attention.head_count_kv")
        self.HD = self.D // self.H
        self.eps = m.dim("attention.layer_norm_rms_epsilon")
        self.base = m.dim("rope.freq_base") if m.has(f"{m.arch}.rope.freq_base") else 10000.0
        self.scale = 1.0
        if m.has(f"{m.arch}.rope.scale_linear"):
            self.scale = m.dim("rope.scale_linear")
        elif m.has(f"{m.arch}.rope.scaling.type") and m.dim("rope.scaling.type") == "linear":
            self.scale = m.dim("rope.scaling.factor")
        if self.HKV != self.H:
            raise NotImplementedError("this teaching rebuild assumes one key/value head per query head (no GQA)")
        self.inv_freq = self.base ** (-np.arange(0, self.HD, 2, dtype=np.float64) / self.HD)
        self.E = m.dequant("token_embd.weight")
        self.W_out = m.dequant("output.weight")
        self.out_norm = m.dequant("output_norm.weight")

    def rope(self, X, pos):
        """Rotate adjacent pairs (0,1), (2,3), ... by a position-dependent angle (llama.cpp 'normal' layout)."""
        ang = (np.asarray(pos, dtype=np.float64) / self.scale)[:, None] * self.inv_freq[None, :]
        c = np.cos(ang)[:, None, :].astype(np.float32)
        s = np.sin(ang)[:, None, :].astype(np.float32)
        x, y = X[..., 0::2], X[..., 1::2]
        out = np.empty_like(X)
        out[..., 0::2] = x * c - y * s
        out[..., 1::2] = x * s + y * c
        return out

    def forward(self, ids, cache, start, capture=None):
        """Run tokens `ids`, at positions start.., through the model. `cache` holds [K, V] per layer."""
        m, N = self.m, len(ids)
        pos = np.arange(start, start + N)
        X = self.E[ids].copy()
        for l in range(self.L):
            p = f"blk.{l}."
            Xn = rms_norm(X, m.dequant(p + "attn_norm.weight"), self.eps)
            Q = (Xn @ m.dequant(p + "attn_q.weight").T).reshape(N, self.H, self.HD)
            K = (Xn @ m.dequant(p + "attn_k.weight").T).reshape(N, self.HKV, self.HD)
            V = (Xn @ m.dequant(p + "attn_v.weight").T).reshape(N, self.HKV, self.HD)
            Q, K = self.rope(Q, pos), self.rope(K, pos)
            if cache[l] is None:
                cache[l] = [K, V]
            else:
                cache[l] = [np.concatenate([cache[l][0], K]), np.concatenate([cache[l][1], V])]
            K_all, V_all = cache[l]
            scores = np.einsum("nhd,mhd->hnm", Q, K_all) / math.sqrt(self.HD)
            future = np.arange(K_all.shape[0])[None, :] > pos[:, None]          # the causal mask
            scores = np.where(future[None], -np.inf, scores)
            A = softmax(scores)
            if capture is not None:
                capture.append(A)
            att = np.einsum("hnm,mhd->nhd", A, V_all).reshape(N, self.D)
            X = X + att @ m.dequant(p + "attn_output.weight").T
            Xn = rms_norm(X, m.dequant(p + "ffn_norm.weight"), self.eps)
            gate = Xn @ m.dequant(p + "ffn_gate.weight").T
            up = Xn @ m.dequant(p + "ffn_up.weight").T
            X = X + (silu(gate) * up) @ m.dequant(p + "ffn_down.weight").T
        return self.W_out @ rms_norm(X[-1], self.out_norm, self.eps)


def table(names, rows, label):
    print(f"  {label:>10} " + " ".join(f"{n!r:>10}" for n in names))
    for name, row in rows:
        print(f"  {name:>10} " + " ".join(f"{v:10.3f}" if not np.isnan(v) else f"{'.':>10}" for v in row))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--focus", default=" France", help="prompt token whose incoming attention to report")
    ap.add_argument("--append", default=" Paris", help="token appended for the KV-cache test")
    ap.add_argument("--server", help="llama-server URL to compare predictions with")
    args = ap.parse_args()

    m = GGUFModel(args.model)
    rb = Rebuild(m)
    print(f"layers={rb.L} heads={rb.H} kv_heads={rb.HKV} head_dim={rb.HD} rope_base={rb.base} "
          f"rope_linear_scale={rb.scale} context_length={m.dim('context_length')}")

    ids = m.encode(args.prompt)
    names = [m.decode([i]) for i in ids]
    print("prompt tokens:", list(zip(ids, names)))

    cap, cache = [], [None] * rb.L
    t0 = time.time()
    probs = softmax(rb.forward(ids, cache, 0, capture=cap))
    print(f"forward pass: {time.time() - t0:.1f} s")
    top = np.argsort(-probs)[:6]

    print("\n== 1. prediction check ==")
    ref = {}
    if args.server:
        srv = LlamaServer(args.server)
        res = srv.complete(args.prompt, n_predict=1, n_probs=10, temperature=0, post_sampling_probs=False)
        ref = {t["id"]: math.exp(t["logprob"]) for t in res["completion_probabilities"][0]["top_logprobs"]}
    for i in top:
        line = f"  {m.decode([int(i)])!r:10} rebuild {probs[i]:.4f}"
        if args.server:
            line += f"   llama.cpp {ref.get(int(i), float('nan')):.4f}"
        print(line)

    print("\n== 2. attention ==")
    W = np.stack([a[:, -1, :] for a in cap])               # (layers, heads, positions): from the last token
    print("  from the last token, mean over heads, selected layers:")
    table(names, [(f"layer {l}", W[l].mean(0)) for l in (0, 1, 2, 8, 16, 24, rb.L - 1)], "")
    print(f"  mean weight on the first token over all layers and heads: {W[:, :, 0].mean():.3f}")
    if args.focus in names:
        f = names.index(args.focus)
        ranked = sorted(((float(W[l, h, f]), l, h) for l in range(rb.L) for h in range(rb.H)), reverse=True)
        print(f"  heads giving {args.focus!r} more than half their weight: {sum(w > 0.5 for w, _, _ in ranked)} of {rb.L * rb.H}")
        print("  strongest:", [(f"layer {l} head {h}", round(w, 3)) for w, l, h in ranked[:5]])
        _, l, h = ranked[0]
        A = cap[l][h].copy()
        A[np.triu_indices_from(A, 1)] = np.nan
        print(f"  full pattern, layer {l} head {h} (row = looking from, column = looking at):")
        table(names, list(zip(names, A)), "")
    A0 = cap[0].mean(0).copy()
    A0[np.triu_indices_from(A0, 1)] = np.nan
    print("  layer 0, mean over heads:")
    table(names, list(zip(names, A0)), "")

    print("\n== 3. KV cache ==")
    extra = m.encode(args.append)
    assert len(extra) == 1, f"{args.append!r} is not a single token"
    full = rb.forward(ids + extra, [None] * rb.L, 0)
    cached = rb.forward(extra, cache, len(ids))
    for label, lg in (("recompute all", full), ("use the cache", cached)):
        p = softmax(lg)
        t = np.argsort(-p)[:3]
        print(f"  {label:14} " + "  ".join(f"{m.decode([int(i)])!r} {p[i]:.4f}" for i in t))
    print(f"  largest logit difference: {float(np.abs(full - cached).max()):.2e}")
    per_token = 2 * rb.L * rb.HKV * rb.HD * 2
    print(f"  cache per token: 2 x {rb.L} layers x {rb.HKV * rb.HD} numbers x 2 bytes (FP16) = {per_token:,} bytes")
    for n in (1921, 2048, m.dim("context_length")):
        print(f"    {n:6} tokens -> {n * per_token / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
