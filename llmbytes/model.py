"""Read a llama-architecture GGUF file: metadata, tensors, Q8_0 dequantization, and its tokenizer.

Everything here is deliberately plain NumPy so each step can be read and checked by hand.
"""

from __future__ import annotations

import numpy as np
from gguf import GGUFReader

# Q8_0 layout: blocks of 32 weights, each block = 2-byte FP16 scale + 32 signed 8-bit integers.
Q8_0_BLOCK = 32
Q8_0_BYTES = 2 + Q8_0_BLOCK


class GGUFModel:
    def __init__(self, path: str):
        self.path = path
        self.reader = GGUFReader(path)
        self.tensors = {t.name: t for t in self.reader.tensors}
        self._tokenizer = None

    # ---- metadata -------------------------------------------------------------------

    def has(self, key: str) -> bool:
        return key in self.reader.fields

    def meta(self, key: str):
        """One metadata value: a str for string fields, otherwise a number."""
        f = self.reader.fields[key]
        part = f.parts[f.data[0]]
        if f.types and f.types[0].name == "STRING":
            return bytes(part).decode("utf-8")
        return part.tolist()[0]

    def meta_strings(self, key: str) -> list[str]:
        """A metadata array of strings, such as the vocabulary."""
        f = self.reader.fields[key]
        return [bytes(f.parts[i]).decode("utf-8", "replace") for i in f.data]

    @property
    def arch(self) -> str:
        return self.meta("general.architecture")

    def dim(self, name: str):
        """An architecture setting, e.g. dim('block_count') -> llama.block_count."""
        return self.meta(f"{self.arch}.{name}")

    # ---- tensors --------------------------------------------------------------------

    def shape_rows_cols(self, name: str) -> tuple[int, ...]:
        """Shape as rows x columns. GGUF stores it fastest-varying dimension first, so reverse it."""
        return tuple(int(n) for n in reversed(self.tensors[name].shape))

    def raw_bytes(self, name: str) -> np.ndarray:
        return np.asarray(self.tensors[name].data, dtype=np.uint8).reshape(-1)

    def dequant(self, name: str) -> np.ndarray:
        """Return a tensor as float32: a 1-D vector, or a (rows, cols) matrix."""
        t = self.tensors[name]
        kind = t.tensor_type.name
        if kind == "F32":
            return np.asarray(t.data, dtype=np.float32).reshape(-1)
        if kind != "Q8_0":
            raise NotImplementedError(f"{name}: only F32 and Q8_0 are handled here, not {kind}")
        cols = int(t.shape[0])
        rows = int(t.n_elements) // cols
        blocks = np.asarray(t.data, dtype=np.uint8).reshape(rows, cols // Q8_0_BLOCK, Q8_0_BYTES)
        scale = blocks[:, :, :2].copy().view(np.float16).astype(np.float32)   # (rows, nblocks, 1)
        ints = blocks[:, :, 2:].copy().view(np.int8).astype(np.float32)        # (rows, nblocks, 32)
        return (ints * scale).reshape(rows, cols)

    def dequant_row(self, name: str, row: int) -> np.ndarray:
        """One row of a Q8_0 matrix, without dequantizing the whole thing."""
        t = self.tensors[name]
        cols = int(t.shape[0])
        rows = int(t.n_elements) // cols
        blocks = np.asarray(t.data, dtype=np.uint8).reshape(rows, cols // Q8_0_BLOCK, Q8_0_BYTES)[row]
        scale = blocks[:, :2].copy().view(np.float16).astype(np.float32)
        ints = blocks[:, 2:].copy().view(np.int8).astype(np.float32)
        return (ints * scale).reshape(cols)

    # ---- tokenizer ------------------------------------------------------------------

    @property
    def vocab(self) -> list[str]:
        return self.meta_strings("tokenizer.ggml.tokens")

    def tokenizer(self):
        """A byte-level BPE tokenizer rebuilt from the file's own vocabulary and merge list.

        It uses the generic GPT-2 pre-tokenizer, not necessarily the model's own, so a few edge
        cases can split differently from llama.cpp. Lesson 2 shows one; pass --server to the
        lesson 2 script to compare against llama.cpp's tokenizer directly.
        """
        if self._tokenizer is None:
            from tokenizers import Tokenizer, decoders, models, pre_tokenizers

            vocab = {t: i for i, t in enumerate(self.vocab)}
            merges = [tuple(m.split(" ", 1)) for m in self.meta_strings("tokenizer.ggml.merges")]
            tk = Tokenizer(models.BPE(vocab=vocab, merges=merges))
            tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
            tk.decoder = decoders.ByteLevel()
            self._tokenizer = tk
        return self._tokenizer

    def encode(self, text: str) -> list[int]:
        return self.tokenizer().encode(text).ids

    def decode(self, ids) -> str:
        return self.tokenizer().decode(list(ids))


# ---- small math used across lessons ----------------------------------------------------


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    """Divide by the root mean square (last axis), then apply the learned per-element gain."""
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * weight


def silu(z: np.ndarray) -> np.ndarray:
    """z * sigmoid(z)."""
    return z / (1.0 + np.exp(-z))


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=-1, keepdims=True)   # subtracting a constant changes nothing
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
