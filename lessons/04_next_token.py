"""Lesson 4 - The forward pass.

Asks a running llama-server for the next-token probabilities of a few prompts, then measures
prefill (a long prompt processed at once) against decode (tokens generated one at a time).

Start a server first (see README), then:

    uv run python lessons/04_next_token.py --server http://127.0.0.1:8099
"""

import argparse
import math

from llmbytes.server import LlamaServer

PROMPTS = [
    "The capital of France is",
    "The cat sat on the",
    "def fibonacci(n):\n    if n",
    "for (int i = 0; i <",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="http://127.0.0.1:8099")
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()
    srv = LlamaServer(args.server)
    if not srv.health():
        raise SystemExit(f"no llama-server answering at {args.server}")

    for p in PROMPTS:
        res = srv.complete(p, n_predict=1, n_probs=args.top, temperature=0, post_sampling_probs=False)
        cp = res["completion_probabilities"][0]
        print(f"\n{p!r}  ({res['tokens_evaluated']} prompt tokens)")
        total = 0.0
        for t in cp["top_logprobs"]:
            prob = math.exp(t["logprob"])
            total += prob
            print(f"  {t['token']!r:12} id {t['id']:6}  p = {prob:.4f}   logprob = {t['logprob']:.4f}")
        print(f"  top-{args.top} total = {total:.4f}")

    print("\n== decode: generating one token at a time ==")
    res = srv.complete("Write a Python function that checks whether a number is prime.\n", n_predict=128, temperature=0)
    tm = res["timings"]
    print(f"  {tm['predicted_n']} tokens at {tm['predicted_per_second']:.1f} tokens/s")

    print("\n== prefill: a long prompt processed at once ==")
    long_prompt = "The quick brown fox jumps over the lazy dog. " * 160
    print("  prompt tokens:", len(srv.tokenize(long_prompt)))
    for _ in range(2):
        tm = srv.complete(long_prompt, n_predict=64, temperature=0, ignore_eos=True)["timings"]
        print(f"  prefill {tm['prompt_n']} tokens in {tm['prompt_ms'] / 1e3:.2f} s = {tm['prompt_per_second']:.0f} tokens/s;"
              f"  then decode {tm['predicted_per_second']:.1f} tokens/s")


if __name__ == "__main__":
    main()
