"""Lesson 2 - Tokens and embeddings.

Bytes and UTF-8, the vocabulary as a string table, byte-level BPE tokenizations, the embedding
table as a row lookup, cosine similarity between rows, and nearest neighbours.

    uv run python lessons/02_tokens_and_embeddings.py MODEL.gguf
    uv run python lessons/02_tokens_and_embeddings.py MODEL.gguf --server http://127.0.0.1:8099

With --server, every tokenization is also checked against llama.cpp's own tokenizer.
"""

import argparse

import numpy as np
from tokenizers.pre_tokenizers import ByteLevel

from llmbytes import GGUFModel, cosine
from llmbytes.server import LlamaServer

SAMPLES = [
    "The cat sat on the mat.",
    "def foo(x):\n    return x * 2",
    "antidisestablishmentarianism",
    " configured the DGX Spark.",
    "  indented",
]
BYTE_SAMPLES = ["A", "é", "猫", "\U0001F999", "ᚠᚢᚦ"]   # A, e-acute, cat (CJK), llama emoji, runes
PAIRS = [(" Monday", " Tuesday"), (" int", " float"), (" int", " string"), (" cat", " dog"),
         (" def", " function"), (" cat", " int"), (" cat", " return"), (" Monday", " return")]
NEIGHBOURS = [" Monday", " int", " def", " cat"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--server", help="llama-server URL, to cross-check tokenizations")
    args = ap.parse_args()

    m = GGUFModel(args.model)
    vocab = m.vocab
    srv = LlamaServer(args.server) if args.server else None
    if srv and not srv.health():
        raise SystemExit(f"no llama-server answering at {args.server}")

    print("== bytes ==")
    for s in ["the cat"] + BYTE_SAMPLES:
        print(f"  {s!r:12} UTF-8 bytes: {s.encode('utf-8').hex(' ')}")

    print("\n== the vocabulary ==")
    print("  size:", len(vocab), " merges:", len(m.meta_strings("tokenizer.ggml.merges")))
    for i in (0, 5000, 5939, 20000, 31000):
        print(f"  id {i:5} -> {vocab[i]!r}")
    index = {t: i for i, t in enumerate(vocab)}
    alphabet = ByteLevel.alphabet()
    print(f"  byte-level stand-ins: {len(alphabet)}, missing from vocab: {sum(c not in index for c in alphabet)}")
    for ch, meaning in (("Ġ", "space 0x20"), ("Ċ", "newline 0x0A"), ("ĉ", "tab 0x09")):
        print(f"  {meaning:13} is shown as {ch!r}, vocab id {index.get(ch)}")
    for key in ("tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id"):
        if m.has(key):
            print(f"  {key} = {m.meta(key)} {vocab[m.meta(key)]!r}")

    print("\n== tokenizations (rebuilt tokenizer" + (" vs llama.cpp" if srv else "") + ") ==")
    for s in SAMPLES + BYTE_SAMPLES:
        ids = m.encode(s)
        pieces = m.tokenizer().encode(s).tokens
        line = f"  {s!r}\n      ids {ids}\n      pieces {pieces}\n      round-trip {m.decode(ids) == s}"
        if srv:
            ref = srv.tokenize(s)
            line += "\n      llama.cpp " + ("MATCH" if ref == ids else f"DIFFERS: {ref}")
        print(line)

    print("\n== embeddings: one row per token ==")
    cat = m.encode(" cat")[0]
    row = m.dequant_row("token_embd.weight", cat)
    print(f"  ' cat' = id {cat}, row of {row.size} numbers, first 8: {np.round(row[:8], 4).tolist()}")

    def vec(word):
        ids = m.encode(word)
        if len(ids) != 1:
            return None
        return m.dequant_row("token_embd.weight", ids[0])

    print("\n== cosine similarity between embedding rows ==")
    for a, b in PAIRS:
        va, vb = vec(a), vec(b)
        print(f"  {a!r:10} . {b!r:10} " + (f"{cosine(va, vb):+.3f}" if va is not None and vb is not None else "(multi-token)"))

    print("\n== nearest neighbours over the whole vocabulary ==")
    E = m.dequant("token_embd.weight")
    norms = np.linalg.norm(E, axis=1) + 1e-9
    for w in NEIGHBOURS:
        q = E[m.encode(w)[0]]
        sims = E @ q / (norms * np.linalg.norm(q))
        top = np.argsort(-sims)[1:9]
        print(f"  {w!r:10} -> {[m.decode([int(i)]) for i in top]}")


if __name__ == "__main__":
    main()
