"""Lesson 5 - Generating text.

The generation loop watched step by step, sampling settings applied to one real distribution
(and re-derived by hand), sampling with different seeds, stopping on end-of-sequence, and
what a chat template turns into.

Start a server first (see README), then:

    uv run python lessons/05_sampling.py --server http://127.0.0.1:8099
"""

import argparse
import math

from llmbytes.server import LlamaServer

FIB = "def fibonacci(n):\n    if n"
NEUTRAL = {"temperature": 1.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0}
SETTINGS = [("raw", {}), ("temperature 0.5", {"temperature": 0.5}), ("temperature 1.5", {"temperature": 1.5}),
            ("top_k 3", {"top_k": 3}), ("top_p 0.8", {"top_p": 0.8}), ("min_p 0.3", {"min_p": 0.3})]


def by_hand(raw, setting):
    """Apply one setting to the raw top probabilities, the way the lesson explains it."""
    probs = list(raw)
    if "temperature" in setting:
        probs = [p ** (1 / setting["temperature"]) for p in probs]
    elif "top_k" in setting:
        probs = probs[: setting["top_k"]]
    elif "top_p" in setting:
        keep, run = [], 0.0
        for p in probs:
            keep.append(p)
            run += p
            if run >= setting["top_p"]:
                break
        probs = keep
    elif "min_p" in setting:
        probs = [p for p in probs if p >= setting["min_p"] * probs[0]]
    return probs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="http://127.0.0.1:8099")
    args = ap.parse_args()
    srv = LlamaServer(args.server)
    if not srv.health():
        raise SystemExit(f"no llama-server answering at {args.server}")

    print("== 1. the loop, greedy, step by step ==")
    res = srv.complete(FIB, n_predict=24, n_probs=3, temperature=0, post_sampling_probs=False)
    for i, cp in enumerate(res["completion_probabilities"], 1):
        alts = ", ".join(f"{t['token']!r} {math.exp(t['logprob']):.2f}" for t in cp["top_logprobs"])
        print(f"  step {i:2}: chose {cp['token']!r:12} | top-3: {alts}")
    print("  text:", repr(FIB + res["content"]), " stop:", res["stop_type"])

    print("\n== 2. one distribution under different settings (server, then by hand) ==")
    print("  (by hand uses only the top 200 tokens, so temperatures above 1 read slightly high)")
    # The full-vocabulary raw distribution, for the by-hand temperature check.
    full = srv.complete(FIB, n_predict=1, n_probs=200, post_sampling_probs=True, seed=1, **NEUTRAL)
    raw_all = [t["prob"] for t in full["completion_probabilities"][0]["top_probs"]]
    for name, s in SETTINGS:
        body = dict(NEUTRAL, **s)
        tops = srv.complete(FIB, n_predict=1, n_probs=6, post_sampling_probs=True, seed=1, **body)
        tops = tops["completion_probabilities"][0]["top_probs"]
        mine = by_hand(raw_all, s)
        total = sum(mine)
        mine = [p / total for p in mine][: len(tops)]
        print(f"  {name:16} server: " + "  ".join(f"{t['token']!r}:{t['prob']:.3f}" for t in tops))
        print(f"  {'':16} by hand: " + "  ".join(f"{p:.3f}" for p in mine) + f"   ({len(by_hand(raw_all, s)) if 'temperature' not in s else 'all'} kept)")

    print("\n== 3. sampling at temperature 0.8 with five seeds ==")
    for seed in (1, 2, 3, 4, 5):
        print(f"  seed {seed}: {srv.complete(FIB, n_predict=14, temperature=0.8, seed=seed)['content']!r}")
    again = {srv.complete(FIB, n_predict=14, temperature=0.8, seed=3)["content"] for _ in range(2)}
    print("  seed 3 twice more gives", "the same text" if len(again) == 1 else f"DIFFERENT texts: {again}")

    print("\n== 4. stopping on end-of-sequence ==")
    r = srv.complete("### Instruction:\nWrite a Python function that returns the square of x.\n### Response:\n",
                     n_predict=200, temperature=0)
    print(f"  stop_type={r['stop_type']} after {r['tokens_predicted']} tokens")
    print("  " + r["content"].replace("\n", "\n  "))

    print("== 5. what a chat request becomes ==")
    t = srv.post("/apply-template", {"messages": [{"role": "system", "content": "You are a helpful assistant."},
                                                  {"role": "user", "content": "What is 2+2?"}]})
    print("  template output:", repr(t["prompt"]))
    pieces = srv.post("/tokenize", {"content": t["prompt"], "add_special": False, "with_pieces": True})["tokens"]
    print("  first tokens:", [p["piece"] for p in pieces[:9]])


if __name__ == "__main__":
    main()
