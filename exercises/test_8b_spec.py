"""Exercise 8b, the capstone: model speculative decoding, in exercises/my_spec.py."""

import math

from _exercise import learner

mine = learner("8b", "spec")
tokens_per_step = mine.tokens_per_step
speedup = mine.speedup


def test_no_guess_accepted():
    assert tokens_per_step(0.0, 5) == 1.0


def test_every_guess_accepted():
    assert tokens_per_step(1.0, 5) == 6.0


def test_closed_form():
    p, k = 0.8, 4
    assert math.isclose(tokens_per_step(p, k), (1 - p ** (k + 1)) / (1 - p))


def test_free_draft_is_pure_gain():
    assert math.isclose(speedup(0.8, 4, 0.0), tokens_per_step(0.8, 4))


def test_a_bad_draft_loses():
    assert speedup(0.2, 5, 0.2) < 1.0
