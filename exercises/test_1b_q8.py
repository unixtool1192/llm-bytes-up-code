"""Exercise 1b, lesson 1: write the Q8_0 decoder, in exercises/my_q8.py."""

import numpy as np

from _exercise import learner

mine = learner("1b", "q8")
dequant_q8_0 = mine.dequant_q8_0

BLOCK = bytes.fromhex(
    "6311" "0081 0807 f40d fbfb fdf5 09fd f500 fcf9 0105 fb01 160c 03fe 09f1 f8fd 0f11 fcf0".replace(" ", "")
)


def test_length():
    assert len(BLOCK) == 34
    assert len(dequant_q8_0(BLOCK)) == 32


def test_first_weights():
    w = dequant_q8_0(BLOCK)
    assert [round(float(x), 4) for x in w[:5]] == [0.0, -0.0835, 0.0053, 0.0046, -0.0079]
