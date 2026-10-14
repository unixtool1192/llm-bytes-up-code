"""Exercise 8b: speculative decoding, modelled. Graded by exercises/test_8b_spec.py."""


def tokens_per_step(p, k):
    ...  # 1 (the real model's own token) + p + p² + ... + p^k


def speedup(p, k, draft_cost):
    ...  # tokens per step ÷ (1.0 for the real model's pass + k draft passes)
