"""Exercise 4b's solution, as lesson 4 gives it."""

import math

import numpy as np


def softmax(logits):
    z = np.exp(logits - np.max(logits))
    return z / z.sum()


def logprob_to_p(lp):
    return math.exp(lp)
