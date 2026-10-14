"""Exercise 6b: one head of attention. Graded by exercises/test_6b_attention.py."""

import numpy as np


def attention(Q, K, V):
    """One head. Q, K, V: N rows (positions) x d columns. Returns (weights N x N, output N x d)."""
    ...  # scores = Q·Kᵀ / √d, mask the future, softmax each row, blend the values
