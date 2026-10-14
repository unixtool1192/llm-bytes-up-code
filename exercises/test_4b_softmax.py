"""Exercise 4b, lesson 4: write softmax, in exercises/my_softmax.py.

The first test is lesson 4's worked example. The last is the server's real logprob for " Paris",
from the recorded run.
"""

import math

import numpy as np

from _exercise import learner

mine = learner("4b", "softmax")
softmax = mine.softmax
logprob_to_p = mine.logprob_to_p


def test_worked_example():
    p = softmax(np.array([2.0, 1.0, 0.1]))
    assert np.allclose(p, [0.659, 0.242, 0.099], atol=0.001)
    assert math.isclose(p.sum(), 1.0)


def test_only_the_gaps_matter():
    a = softmax(np.array([2.0, 1.0, 0.1]))
    b = softmax(np.array([1002.0, 1001.0, 1000.1]))
    assert np.allclose(a, b)


def test_paris():
    assert round(logprob_to_p(-0.0756), 4) == 0.9272
