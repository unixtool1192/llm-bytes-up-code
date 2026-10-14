"""Exercise 3b, lesson 3: write RMS and SiLU, in exercises/my_ffn.py.

The last test reads " cat"'s row from the model file, so it needs MODEL set; without it, it skips.
"""

import math
import os

import numpy as np

from llmbytes import GGUFModel

from _exercise import learner, needs_model

mine = learner("3b", "ffn")
rms = mine.rms
silu = mine.silu


def test_rms_worked_example():
    assert rms(np.array([4.0, -2.0, 0.0, 4.0])) == 3.0


def test_silu_table():
    z = np.array([-4.0, -1.0, 0.0, 1.0, 2.0])
    assert np.round(silu(z), 3).tolist() == [-0.072, -0.269, 0.0, 0.731, 1.762]


@needs_model
def test_rms_of_cat():
    m = GGUFModel(os.environ["MODEL"])
    x = m.dequant_row("token_embd.weight", m.encode(" cat")[0])
    assert round(rms(x), 4) == 0.1708
