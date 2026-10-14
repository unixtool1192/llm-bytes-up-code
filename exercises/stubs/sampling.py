"""Exercise 5b: the sampling settings. Graded by exercises/test_5b_sampling.py.

The tests call each function on RAW, the six most likely next tokens from lesson 4's run,
sorted from most to least likely.
"""


def rescale(probs):
    total = sum(probs)
    return [p / total for p in probs]


def temperature(probs, t):
    ...  # raise each to the power 1/t, then rescale


def top_k(probs, k):
    ...  # keep the k most likely (RAW is already sorted), then rescale


def top_p(probs, p):
    ...  # keep the smallest top group whose total reaches p, then rescale


def min_p(probs, m):
    ...  # keep every probability at least m times the top one, then rescale
