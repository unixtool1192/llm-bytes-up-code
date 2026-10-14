"""Exercise 2b, lesson 2: write cosine similarity, in exercises/my_cosine.py.

The last test reads two real rows from the model file, so it needs MODEL set; without it, it skips.
"""

import math
import os

import numpy as np

from llmbytes import GGUFModel

from _exercise import learner, needs_model

mine = learner("2b", "cosine")
length = mine.length
cosine = mine.cosine


def test_length_of_3_4():
    assert length(np.array([3.0, 4.0])) == 5.0


def test_directions():
    x = np.array([1.0, 2.0, 3.0])
    assert math.isclose(cosine(x, 2 * x), 1.0)
    assert math.isclose(cosine(x, -x), -1.0)
    assert abs(cosine(np.array([1.0, 0.0]), np.array([0.0, 1.0]))) < 1e-12


@needs_model
def test_real_rows():
    m = GGUFModel(os.environ["MODEL"])

    def row(word):
        return m.dequant_row("token_embd.weight", m.encode(word)[0])

    assert round(cosine(row(" cat"), row(" dog")), 3) == 0.232
    assert round(cosine(row(" Monday"), row(" Tuesday")), 3) == 0.410
