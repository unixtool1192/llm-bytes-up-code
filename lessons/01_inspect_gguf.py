"""Lesson 1 - What a model is on disk.

Opens a GGUF file and prints its metadata, its tensor directory (shapes as rows x columns),
the total parameter count, and the first Q8_0 block of one matrix decoded by hand.

    uv run python lessons/01_inspect_gguf.py MODEL.gguf
"""

import argparse
import os
from collections import Counter

import numpy as np

from llmbytes import GGUFModel


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="path to a llama-architecture GGUF file (the course uses Q8_0)")
    ap.add_argument("--tensor", default="blk.0.attn_q.weight", help="matrix whose first bytes to decode")
    args = ap.parse_args()

    m = GGUFModel(args.model)
    print("file bytes:", f"{os.path.getsize(args.model):,}")
    print("architecture:", m.arch)
    for key in ("context_length", "embedding_length", "block_count", "feed_forward_length",
                "attention.head_count", "attention.head_count_kv"):
        print(f"  {m.arch}.{key} = {m.dim(key)}")

    print(f"\ntensor directory: {len(m.tensors)} tensors")
    print(f"  {'name':28} {'shape (rows x cols)':22} {'type':6} {'numbers':>14} {'byte offset':>14}")
    names = list(m.tensors)
    shown = names[:10] + ["..."] + names[-2:]
    for n in shown:
        if n == "...":
            print("  ...  (the nine blk.N tensors repeat for every block)")
            continue
        t = m.tensors[n]
        shape = " x ".join(str(s) for s in m.shape_rows_cols(n))
        print(f"  {n:28} {shape:22} {t.tensor_type.name:6} {int(t.n_elements):>14,} {int(t.data_offset):>14,}")

    groups = Counter(n.split(".")[0] for n in names)
    print("\ntensor name prefixes:", dict(groups))
    total = sum(int(t.n_elements) for t in m.tensors.values())
    print(f"total parameters: {total:,}")

    # Parameter count by hand for one block, compared with the file.
    blk = sum(int(m.tensors[n].n_elements) for n in names if n.startswith("blk.0."))
    nblk = m.dim("block_count")
    rest = total - blk * nblk
    print(f"  one block: {blk:,} x {nblk} blocks = {blk * nblk:,}; embedding + output + final norm: {rest:,}")

    print("\ndtype -> size of this model:")
    for name, bpp in (("FP32", 4), ("BF16/FP16", 2), ("Q8_0", 34 / 32), ("~4-bit", 0.5)):
        print(f"  {name:10} {bpp:6.4f} bytes/param -> {total * bpp / 1e9:5.2f} GB")

    raw = m.raw_bytes(args.tensor)[:34]
    print(f"\nfirst 34 bytes of {args.tensor}:")
    print("  " + " ".join(f"{b:02x}" for b in raw))
    scale = raw[:2].copy().view(np.float16)[0]
    ints = raw[2:].copy().view(np.int8)
    print(f"  scale (FP16 0x{raw[1]:02x}{raw[0]:02x}) = {float(scale):.6f}")
    print("  first 5 int8:", ints[:5].tolist())
    print("  first 5 weights = scale x int8:", [round(float(scale) * int(i), 4) for i in ints[:5]])
    check = m.dequant(args.tensor)[0, :5]
    print("  same via dequant():", [round(float(v), 4) for v in check])


if __name__ == "__main__":
    main()
