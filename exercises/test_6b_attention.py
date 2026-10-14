"""Exercise 6b, lesson 6: write one head of attention, in exercises/my_attention.py.

It needs no model: the tests use random vectors, five positions of 128 numbers, the size of one head.
"""

import numpy as np

from _exercise import learner

mine = learner("6b", "attention")
attention = mine.attention

RNG = np.random.default_rng(0)
Q, K, V = (RNG.standard_normal((5, 128)) for _ in range(3))


def test_rows_are_weights():
    w, _ = attention(Q, K, V)
    assert np.allclose(w.sum(axis=1), 1.0)
    assert (w >= 0).all()


def test_no_looking_ahead():
    w, _ = attention(Q, K, V)
    assert np.all(w[np.triu_indices(5, k=1)] == 0.0)


def test_first_position_sees_only_itself():
    w, out = attention(Q, K, V)
    assert w[0, 0] == 1.0
    assert np.allclose(out[0], V[0])


def test_a_matching_key_wins():
    K2 = K.copy()
    K2[3] = 20 * Q[4]                                   # position 3 offers exactly what 4 wants
    w, _ = attention(Q, K2, V)
    assert w[4].argmax() == 3 and w[4, 3] > 0.9
