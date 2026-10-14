"""Exercise 6b's solution, as lesson 6 gives it."""

import numpy as np


def attention(Q, K, V):
    n, d = Q.shape
    scores = Q @ K.T / np.sqrt(d)
    scores[np.triu_indices(n, k=1)] = -np.inf          # causal mask: no looking ahead
    scores -= scores.max(axis=1, keepdims=True)
    w = np.exp(scores)
    w /= w.sum(axis=1, keepdims=True)
    return w, w @ V
