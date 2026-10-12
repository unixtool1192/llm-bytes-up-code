"""Capstone - Why decode is memory bound: batching, measured.

Sends 1, 2, 4, 8 and 16 generation requests to llama-server *at the same time* and records how
fast each user's text arrives and how many tokens the server produces in total. Below the
memory-to-compute crossover, adding users should barely slow anyone down, because every user's
token shares the same read of the weights.

The server needs enough parallel slots, so start it with, for example:

    LESSON_PARALLEL=16 LESSON_CTX=16384 scripts/start_server.sh MODEL.gguf
    uv run python lessons/08_capstone_batching.py --server http://127.0.0.1:8099
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor

from llmbytes.server import LlamaServer

TOPICS = ["a binary search", "a linked list", "quicksort", "a hash table", "a stack", "a queue",
          "merge sort", "a binary tree", "Dijkstra's algorithm", "a trie", "a heap", "bubble sort",
          "a ring buffer", "a bloom filter", "an LRU cache", "a graph traversal"]


def one_request(srv, i, n_predict):
    prompt = f"### Instruction:\nWrite {TOPICS[i % len(TOPICS)]} in C, with comments.\n### Response:\n"
    res = srv.complete(prompt, n_predict=n_predict, temperature=0, ignore_eos=True)
    return res["timings"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="http://127.0.0.1:8099")
    ap.add_argument("--tokens", type=int, default=256, help="tokens each user generates")
    ap.add_argument("--users", default="1,2,4,8,16")
    args = ap.parse_args()
    srv = LlamaServer(args.server, timeout=900)
    if not srv.health():
        raise SystemExit(f"no llama-server answering at {args.server}")

    one_request(srv, 0, 16)   # warm-up
    print(f"each user generates {args.tokens} tokens; all users start at the same moment\n")
    print(f"  {'users':>5} {'wall time':>10} {'per-user tok/s':>16} {'total tok/s':>12} {'vs 1 user':>10}")
    base = None
    for n in [int(u) for u in args.users.split(",")]:
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n) as pool:
            timings = list(pool.map(lambda i: one_request(srv, i, args.tokens), range(n)))
        wall = time.perf_counter() - t0
        per_user = sum(t["predicted_per_second"] for t in timings) / n
        total = sum(t["predicted_n"] for t in timings) / wall
        base = base or total
        print(f"  {n:>5} {wall:>9.1f}s {per_user:>16.1f} {total:>12.1f} {total / base:>9.1f}x")


if __name__ == "__main__":
    main()
