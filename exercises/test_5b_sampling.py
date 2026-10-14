"""Exercise 5b, lesson 5: write the sampling settings, in exercises/my_sampling.py.

No server is needed: the six probabilities are the raw row from lesson 4's run.
"""

import pytest

from _exercise import learner

mine = learner("5b", "sampling")
temperature = mine.temperature
top_k = mine.top_k
top_p = mine.top_p
min_p = mine.min_p

# The six most likely first tokens after "def fibonacci(n):\n    if n", from lesson 4's run:
# ' == ', ' <= ', '<', ' < ', '==', ' in'. Together they hold 97.8% of the probability.
RAW = [0.3075, 0.2792, 0.1455, 0.1219, 0.0999, 0.0238]


def r3(probs):
    return [round(p, 3) for p in probs]


def test_temperature_half():
    # Only six tokens, not the whole vocabulary, so allow for the missing tail.
    assert abs(temperature(RAW, 0.5)[0] - 0.431) < 0.002


def test_top_k_3():
    assert r3(top_k(RAW, 3)) == [0.420, 0.381, 0.199]


def test_top_p_08():
    assert r3(top_p(RAW, 0.8)) == [0.360, 0.327, 0.170, 0.143]


def test_min_p_03():
    assert r3(min_p(RAW, 0.3)) == [0.322, 0.293, 0.153, 0.128, 0.105]


@pytest.mark.parametrize("f", [lambda p: top_k(p, 2), lambda p: top_p(p, 0.5), lambda p: min_p(p, 0.5)])
def test_three_settings_one_answer(f):
    assert r3(f(RAW)) == [0.524, 0.476]
