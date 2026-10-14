"""Exercise 2b's solution, as lesson 2 gives it."""

import math

import numpy as np


def length(v):
    return math.sqrt(float(np.dot(v, v)))


def cosine(a, b):
    return float(np.dot(a, b)) / (length(a) * length(b))
