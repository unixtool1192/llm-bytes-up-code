"""Exercise 5b's solution, as lesson 5 gives it."""


def rescale(probs):
    total = sum(probs)
    return [p / total for p in probs]


def temperature(probs, t):
    return rescale([p ** (1 / t) for p in probs])


def top_k(probs, k):
    return rescale(probs[:k])


def top_p(probs, p):
    kept, run = [], 0.0
    for q in probs:
        kept.append(q)
        run += q
        if run >= p:
            break
    return rescale(kept)


def min_p(probs, m):
    return rescale([q for q in probs if q >= m * probs[0]])
