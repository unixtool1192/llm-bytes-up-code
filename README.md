# llm-bytes-up-code

Companion code for the course **LLMs from the Bytes Up**. The course explains how a large language model works by **opening a real model file and reading the numbers in it**, rather than drawing diagrams. Every number a lesson quotes as *observed* comes from one of these scripts.

The lessons are written for engineers who are fluent in C and systems but new to machine learning, so the code stays plain: NumPy, the `gguf` reader, and nothing that hides the arithmetic.

## What each script shows

| Script | Lesson | What it does | Needs a server? |
|---|---|---|---|
| `lessons/01_inspect_gguf.py` | 1 — What a model is on disk | Metadata, the tensor directory (shapes as rows × columns), the parameter count, and one Q8_0 block decoded by hand | no |
| `lessons/02_tokens_and_embeddings.py` | 2 — Tokens and embeddings | Bytes and UTF-8, the vocabulary, BPE tokenizations, embedding rows, cosine similarity, nearest neighbours | optional, to cross-check tokenizations |
| `lessons/03_ffn_by_hand.py` | 3 — A layer is matrix multiplication | One token through block 0's feed-forward network: RMS, matvec vs a plain loop, SiLU, residual and cosine, FLOPs, matvec vs matmul timing | no |
| `lessons/04_next_token.py` | 4 — The forward pass | Next-token probabilities for a few prompts; prefill vs decode speed | **yes** |
| `lessons/05_sampling.py` | 5 — Generating text | The generation loop step by step; temperature, top-k, top-p and min-p applied to one real distribution and re-derived by hand; seeds; end-of-sequence; chat templates | **yes** |
| `lessons/06_attention_rebuild.py` | 6 — Attention and the KV cache | The whole model rebuilt in NumPy: its prediction checked against llama.cpp, every attention weight captured, and the KV cache tested | optional, to compare predictions |
| `lessons/07_hardware/bench.cu` | 7 — The hardware fit | CUDA C: the GPU's device query, a hand-written memory-read kernel, one weight matrix times 1–4,096 tokens (the memory-to-compute crossover), and BF16 tensor cores vs FP32 | no — needs the CUDA toolkit instead |
| `lessons/08_capstone_batching.py` | Capstone — Why decode is memory bound | 1, 2, 4, 8 and 16 users generating at the same time: per-user speed vs total throughput | **yes**, with 16 slots: `LESSON_PARALLEL=16 LESSON_CTX=16384` |

`llmbytes/` holds the shared pieces: reading GGUF metadata, dequantizing Q8_0, rebuilding the tokenizer from the file, and a small client for `llama-server`.

## The model

The course uses one small, plain, dense model throughout: **`deepseek-coder-6.7b-instruct`, as a Q8_0 GGUF** (about 7.2 GB). Any GGUF of that model works. The scripts take its path as an argument, so put it wherever you like.

The scripts handle the llama architecture with F32 and Q8_0 tensors. Other quantizations or architectures will stop with a clear error rather than give wrong numbers. `06_attention_rebuild.py` also assumes one key/value head per query head, which this model has.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone <this repo>
cd llm-bytes-up-code
uv sync
```

## Running

Lessons 1–3, and 6, need only the model file:

```bash
MODEL=/path/to/deepseek-coder-6.7b-instruct.Q8_0.gguf
uv run python lessons/01_inspect_gguf.py "$MODEL"
uv run python lessons/02_tokens_and_embeddings.py "$MODEL"
uv run python lessons/03_ffn_by_hand.py "$MODEL"
uv run python lessons/06_attention_rebuild.py "$MODEL"
```

Lesson 7 is CUDA C. It needs an NVIDIA GPU and the CUDA toolkit (`nvcc`, with cuBLAS), but not the model:

```bash
cd lessons/07_hardware && make && ./bench
```

Lessons 4 and 5 talk to a running llama.cpp `llama-server`. Lessons 2 and 6 can use one as well, for cross-checks. Start a temporary one, bound to localhost:

```bash
LLAMA_SERVER=/path/to/llama.cpp/build/bin/llama-server scripts/start_server.sh "$MODEL"
uv run python lessons/04_next_token.py --server http://127.0.0.1:8099
uv run python lessons/05_sampling.py --server http://127.0.0.1:8099
uv run python lessons/02_tokens_and_embeddings.py "$MODEL" --server http://127.0.0.1:8099
uv run python lessons/06_attention_rebuild.py "$MODEL" --server http://127.0.0.1:8099
kill "$(cat /tmp/llm-bytes-up-server.pid)"
```

The server offloads every layer to the GPU (`-ngl 99`), uses a 4,096-token context, serves one request at a time, and stops itself after 40 minutes if you forget to.

### What to expect

Measured on the course's DGX Spark:

| Script | Wall time | Peak memory |
|---|---|---|
| 01, 02, 03 | about 1 s each | up to 1.4 GB |
| 04, 05 | seconds, mostly the server generating | small (the server holds the model) |
| 06 | about 15 s — three forward passes, each dequantizing ~7 GB of weights | **about 9 GB** |
| 07 (`bench`) | about 2 s | about 2.5 GB of GPU memory at once |
| 08 | about 50 s — five rounds of 256-token generations | small (the server holds the model and 16 users' KV cache, ~8.6 GB) |

- **Speeds depend on the machine.** Lesson 3's matvec timing and lesson 4's tokens/s will differ on other hardware; the *ratios* are the point.
- **Probabilities and tokenizations should match the lessons exactly**, for the same model file and a similar llama.cpp version.
- **Lesson 5's by-hand check uses only the top 200 tokens**, so at temperatures above 1 (which shift weight into the tail) it reads a few thousandths high. The other settings match exactly.

## Recorded runs

`outputs/` holds the output of every script as run for the course, one folder per date, so you can compare your run against the one the lessons quote.

`outputs/2026-10-12/` was produced from a fresh clone of this repository on an NVIDIA DGX Spark (GB10, 128 GB unified memory), with llama.cpp build 10630 (commit `d222767c7`) serving the model on the GPU, and with the `--server` cross-checks enabled for lessons 2 and 6.
