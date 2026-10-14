"""Exercise 1b's solution, as lesson 1 gives it."""

import numpy as np


def dequant_q8_0(block):
    scale = np.frombuffer(block[:2], dtype="<f2")[0]
    q = np.frombuffer(block[2:34], dtype=np.int8)
    return np.float32(scale) * q.astype(np.float32)
