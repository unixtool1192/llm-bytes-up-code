"""Exercise 3b's solution, as lesson 3 gives it."""

import math

import numpy as np


def rms(x):
    return math.sqrt(float(np.mean(x * x)))


def silu(z):
    return z / (1.0 + np.exp(-z))
