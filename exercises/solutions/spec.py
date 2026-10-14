"""Exercise 8b's solution, as the capstone gives it."""


def tokens_per_step(p, k):
    expected, chance = 1.0, 1.0
    for _ in range(k):
        chance *= p
        expected += chance
    return expected


def speedup(p, k, draft_cost):
    return tokens_per_step(p, k) / (1.0 + k * draft_cost)
