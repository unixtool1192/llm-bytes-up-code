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
| `lessons/07_hardware/bench_cpu.py` | 7 — The hardware fit, without an NVIDIA GPU | The same measurements on the CPU with NumPy, in the same output format: memory-read bandwidth, one weight matrix times 1–4,096 tokens, and peak FP32 throughput | no |
| `lessons/08_capstone_batching.py` | Capstone — Why decode is memory bound | 1, 2, 4, 8 and 16 users generating at the same time: per-user speed vs total throughput | **yes**, with 16 slots: `LESSON_PARALLEL=16 LESSON_CTX=16384` |

`llmbytes/` holds the shared pieces: reading GGUF metadata, dequantizing Q8_0, rebuilding the tokenizer from the file, and a small client for `llama-server`.

## The model

The course uses one small, plain, dense model throughout: **`deepseek-coder-6.7b-instruct`, as a Q8_0 GGUF** (about 7.2 GB). The scripts take its path as an argument, so put it wherever you like.

The course's runs used TheBloke's conversion, from Hugging Face: [`TheBloke/deepseek-coder-6.7B-instruct-GGUF`](https://huggingface.co/TheBloke/deepseek-coder-6.7B-instruct-GGUF), file `deepseek-coder-6.7b-instruct.Q8_0.gguf` — 7,163,879,648 bytes, SHA-256 `02cd6ce7ccec670cf6d3dd147932f13e584f9e964d5a3297a74b401b658471ae`. Another GGUF of the same model runs the scripts too, but only this file reproduces the lessons' probabilities exactly.

The scripts handle the llama architecture with F32 and Q8_0 tensors. Other quantizations or architectures will stop with a clear error rather than give wrong numbers. `06_attention_rebuild.py` also assumes one key/value head per query head, which this model has.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone <this repo>
cd llm-bytes-up-code
uv sync
```

Then ask what this machine can run:

```bash
uv run llmbytes doctor --model /path/to/deepseek-coder-6.7b-instruct.Q8_0.gguf
```

`doctor` reports your hardware and which lessons are ready. It checks that the model is the course's exact file, finds `llama-server` (from `$LLAMA_SERVER` or your `PATH`), and measures your memory bandwidth. From that bandwidth it predicts lesson 4's generation speed. Hashing the 7 GB model takes up to a minute the first time; the result is cached, and `--quick` skips it. On a GPU machine it measures GPU bandwidth too, once `lessons/07_hardware/bench` is built.

## Checking a run

`llmbytes check` compares a lesson's output with the course's recorded run:

```bash
uv run python lessons/04_next_token.py --server http://127.0.0.1:8099 | tee my_04.txt
uv run llmbytes check 4 my_04.txt
lessons/07_hardware/bench | uv run llmbytes check 7 -
```

- **✓ lines that do not depend on hardware match the reference:** probabilities, token IDs, shapes and similarities. The last printed digit may differ, and so may column spacing.
	- **What this repository's own NumPy code computes** (lessons 1–3, and lesson 6's rebuild) matched the Spark reference to the last printed digit on two x86 CPUs too, an Intel laptop and an Intel workstation. The one exception is a rounding residual in lesson 6's KV-cache test, which `check` compares only by order of magnitude.
	- **What llama.cpp computes** (lessons 4 and 5, and the `llama.cpp` columns of lessons 2 and 6) depends on the llama.cpp backend that ran it. On three CPU-only builds (the Intel laptop, the Intel workstation and the Spark's own Arm CPU), probabilities differed from the Spark's GPU run from the third decimal place, and from each other, so `check` fails those lines there. Token IDs, tokenizations, greedy choices and seeded text still matched. See *Recorded runs*.
- **ℹ measurements:** times and speeds are never compared one to one, because a slower machine is not a wrong one.
- **✓ shape checks**, where a lesson's point is a pattern:
	- lesson 3: matmul beats matvec per token
	- lesson 4: prefill runs far faster than decode
	- lesson 7: time stays flat while memory-bound, then grows with N
	- capstone: total throughput rises with users
- **Lines the reference has because it ran with `--server`** are skipped when your run did not, and `check` says so.

It exits 1 if anything fails.

## Exercises

Each lesson's code exercise is graded by pytest, from `exercises/`. You write the functions; the tests are already here.

| Exercise | Lesson | You write, in | Graded by |
|---|---|---|---|
| 1b | 1 — the Q8_0 decoder | `exercises/my_q8.py` | `exercises/test_1b_q8.py` |
| 2b | 2 — cosine similarity | `exercises/my_cosine.py` | `exercises/test_2b_cosine.py` |
| 3b | 3 — RMS and SiLU | `exercises/my_ffn.py` | `exercises/test_3b_ffn.py` |
| 4b | 4 — softmax | `exercises/my_softmax.py` | `exercises/test_4b_softmax.py` |
| 5b | 5 — the sampling settings | `exercises/my_sampling.py` | `exercises/test_5b_sampling.py` |
| 6b | 6 — one head of attention | `exercises/my_attention.py` | `exercises/test_6b_attention.py` |
| 8b | Capstone — speculative decoding, modelled | `exercises/my_spec.py` | `exercises/test_8b_spec.py` |

```bash
uv run llmbytes exercise 4b                  # creates exercises/my_softmax.py from its stub if missing, then runs the tests
uv run pytest exercises/test_4b_softmax.py   # the same tests, run directly
```

- **`exercises/stubs/`** holds the starting point for each `my_*.py`: the function signatures, with a comment saying what each returns. `llmbytes exercise` copies the stub and never overwrites your file.
- **Your `my_*` files are ignored by git**, as are the lessons' `my_NN.txt` output copies, so `git pull` never conflicts with your work.
- **A test file whose `my_*.py` does not exist yet is skipped**, with a message saying how to create it.
- **2b and 3b each have one test that reads the model file.** It needs `MODEL` set to the model's path, and skips without it.
- **`exercises/solutions/`** holds the lessons' solutions. `LLMBYTES_SOLUTIONS=1 uv run pytest exercises` grades those instead of your files; that is what CI runs.

## Continuous integration

`.gitlab-ci.yml` runs on every push: `tests/` (the tooling's own tests, including `llmbytes check` against every recorded run), then `exercises/` against the solutions. CI has no model file, so the two model-reading exercise tests skip there.

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
./bench --read-only --blocks 1,2,4,8,16,48,96     # the read test only, once per block count
```

On Windows, `make` is not needed: `nvcc -O3 -std=c++17 -arch=native -ccbin "<Visual Studio>\VC\Tools\MSVC\<version>\bin\Hostx64\x64" -o bench.exe bench.cu -lcublas`.

Without an NVIDIA GPU, run the CPU version. It prints the same sections, so the lesson's chart reads either:

```bash
uv run python lessons/07_hardware/bench_cpu.py
uv run python lessons/07_hardware/bench_cpu.py --read-only --threads 1,2,4,all
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

The server offloads every layer to the GPU (`-ngl 99`; set `LESSON_GPU_LAYERS=0` to run on the CPU, or a smaller number to split the model when the GPU's memory is too small), uses a 4,096-token context, serves one request at a time, and stops itself after 40 minutes if you forget to.

The capstone's batching measurement needs a server that takes 16 requests at once:

```bash
LESSON_PARALLEL=16 LESSON_CTX=16384 LLAMA_SERVER=/path/to/llama-server scripts/start_server.sh "$MODEL"
uv run python lessons/08_capstone_batching.py --server http://127.0.0.1:8099
kill "$(cat /tmp/llm-bytes-up-server.pid)"
```

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
- **Tokenizations match the lessons exactly** for the same model file. **Probabilities match exactly when llama.cpp runs on the Spark's GPU**; on each of three CPU-only builds they differed from the third decimal place, each by its own amount (*Recorded runs*).
- **On a CPU, lessons 4 and 6 take minutes, not seconds.** Lesson 4's 1,921-token prefill took 76.1 s on the Intel laptop, 47.9 s on the Intel workstation and 35.8 s on the Spark's Arm CPU, against 0.80 s on its GPU. Lesson 6's forward pass took 63.0 s, 12.7 s and 7.0 s, against 4.7 s in the reference run.
- **Lesson 5's by-hand check uses only the top 200 tokens**, so at temperatures above 1 (which shift weight into the tail) it reads a few thousandths high. The other settings match exactly.

## Recorded runs

`outputs/` holds the output of every script as run for the course, one folder per date, so you can compare your run against the one the lessons quote.

`outputs/2026-10-12/` was produced from a fresh clone of this repository on an NVIDIA DGX Spark (GB10, 128 GB unified memory), with llama.cpp build 10630 (commit `d222767c7`) serving the model on the GPU, and with the `--server` cross-checks enabled for lessons 2 and 6. It is the reference `llmbytes check` compares against. One line is missing from its lesson 5 output: a caveat the script began printing seconds after the run was recorded. `check` ignores note-only lines, so this does not matter.

`outputs/2026-10-13/` holds lesson 7 on three more machines, one folder per machine, to show the same shape on different hardware:

- `laptop-rtx3000ada/`: bench.cu on a discrete laptop GPU (8 GB, 36 SMs)
- `laptop-intel-cpu/`: bench_cpu.py on an Intel Core Ultra 7 165H with DDR5-5600
- `dgx-spark-arm-cpu/`: bench_cpu.py on the Spark's 20 Arm cores

`outputs/2026-10-14/` holds the runs that test the lessons' "any machine" claims off the Spark, with the same model file (SHA-256 checked) and llama.cpp build 10630 (`d222767c7`):

- `laptop-intel-cpu/`: lessons 1–6 on the Intel Core Ultra 7 165H, under Windows 11 in Git Bash, with llama.cpp built CPU-only from source with MSVC (Visual Studio 18 Build Tools, `cmake -B build`) and started by `scripts/start_server.sh` with its defaults. Lessons 2 and 6 used `--server`. Also `07_bench_cpu_read_only.txt`, a `bench_cpu.py --read-only --threads 1,2,4,8,16,all` sweep, four times.
	- **Matched the Spark to the last printed digit:** lessons 1, 2 (every tokenization, including llama.cpp's) and 3, and every number of lesson 6's NumPy rebuild, attention weights included.
	- **Differed:** every probability llama.cpp reported. Lesson 4's `' Paris'` is 0.9242 against the Spark's 0.9272, and `' <'` and `' ?'` swap places. Lesson 5's settings table differs in the third decimal, and lesson 6's `llama.cpp` column likewise. Lesson 5's greedy choices, all five seeded continuations, the end-of-sequence answer and the chat template matched.
	- **Lesson 4's shape check fails here:** prefill ran at 25 tokens/s and decode at 8.4, about 3x per token against the Spark's 72x, and `check` asks for more than 3x.
- `workstation-intel-i7-9700k-cpu/`: lessons 1–6 on an Intel Core i7-9700K desktop (8 cores, no hyper-threading, DDR4-2133), under Linux, with llama.cpp built CPU-only with gcc 14 (`cmake -B build`, in a Debian 13 container) and started by `scripts/start_server.sh` with its defaults. Lessons 2 and 6 used `--server`. Everything ran at `nice -n 19` on a host whose other services kept running. Also `07_bench_cpu_read_only.txt`, the same thread sweep as the laptop's, four times.
	- **Matched the Spark to the last printed digit:** lessons 1, 2 and 3, and lesson 6's rebuild, apart from the KV-cache test's rounding residual (1.05e-05 against 1.62e-05).
	- **Differed:** every probability llama.cpp reported, from the Spark's and from the laptop's. Lesson 5's greedy choices, seeded continuations, end-of-sequence answer and chat template matched.
	- **Lesson 4's shape check passes:** prefill 40 tokens/s, decode 3.5, about 11x.
- `dgx-spark-arm-cpu/04_next_token.txt`, `05_sampling.txt` and `06_attention_rebuild.txt`: lessons 4–6 on the Spark itself, with llama.cpp built CPU-only with gcc (`cmake -B build`) and the server started with `LESSON_GPU_LAYERS=0`. It logged "no usable GPU found", and `nvidia-smi` listed only the two resident servers while it ran. Lesson 6 used `--server`; the server ran at `nice -n 19`, beside two resident model servers on the GPU.
	- **Differed from the Spark's own GPU run** in every llama.cpp probability, and from both x86 CPUs. Lesson 5's greedy choices, seeded continuations, end-of-sequence answer and chat template matched; lesson 6's rebuild matched exactly, residual included.
	- **Lesson 4's shape check passes:** prefill 54 tokens/s, decode 11.0, about 5x.
- `laptop-rtx3000ada/07_bench_read_only.txt`: `bench --read-only --blocks 1,2,4,8,16,48,96` on the laptop GPU. Bandwidth stops rising at 16 blocks (217 GB/s; 219 at 48 and 96), fewer than its 36 SMs.
- `dgx-spark/07_bench_read_only.txt`: the same sweep with 24 added, on the Spark's GPU, while two resident model servers were loaded and idle. 225 GB/s at 16 blocks, 245 at 24, 255 from 48, against 48 SMs.
- `dgx-spark-arm-cpu/07_bench_cpu_read_only.txt`: the thread sweep on the Spark's Arm CPU, four times. Three runs rose smoothly to about 114 GB/s by 8 threads; in the fourth, 2 and 4 threads read only 45.4 and 47.8 GB/s, then 8 threads reached 113.0.
- `workstation-intel-i7-9700k-cpu/07_bench_cpu_read_only.txt`: the thread sweep on the workstation's CPU, four times. Bandwidth peaked at 4 threads, 28.9–29.1 GB/s, and settled at about 27.5 from 8; one thread read 21.2–21.6.
- **Not run:** llama.cpp on the workstation's GPU (an 8 GB GeForce GTX 1070). Docker on that host had no GPU access, and fixing that is left for later.

### Four llama.cpp backends, one model file

Lessons 4–6 on every machine above, with the same model file and llama.cpp build 10630. The Spark's GPU run is the reference.

| Machine | llama.cpp ran on | Lesson 4 `' Paris'` | Lesson 4 `' == '` | Lesson 5 raw `' == '` | Lesson 6 `llama.cpp` `' Paris'` (rebuild) |
|---|---|---|---|---|---|
| DGX Spark, reference | GPU, Linux | 0.9272 | 0.3075 | 0.307 | 0.9272 (0.9260) |
| Intel laptop | x86 CPU, Windows, MSVC | 0.9242 | 0.3095 | 0.310 | 0.9242 (0.9260) |
| Intel workstation | x86 CPU, Linux, gcc | 0.9260 | 0.3119 | 0.312 | 0.9260 (0.9260) |
| DGX Spark | Arm CPU, Linux, gcc | 0.9233 | 0.3090 | 0.309 | 0.9233 (0.9260) |
| Intel workstation | GPU | not run: Docker had no GPU access (to be fixed later) | | | |

| Machine | Decode (tokens/s) | Prefill (tokens/s) | Prefill ÷ decode | Decode × 7.16 GB | CPU read bandwidth, all threads |
|---|---|---|---|---|---|
| DGX Spark GPU, reference | 33.7 | 2,414 | 72x | 241 GB/s | (GPU: 255 GB/s, `dgx-spark/07_bench_read_only.txt`) |
| Intel laptop CPU | 8.4 | 25 | 3x | 60 GB/s | 61.3 GB/s |
| Intel workstation CPU | 3.5 | 40 | 11x | 25 GB/s | 27.5 GB/s |
| DGX Spark Arm CPU | 11.0 | 54 | 5x | 79 GB/s | 114.0 GB/s |

| CPU read bandwidth (GB/s), run 1 | 1 | 2 | 4 | 8 | 16 | all |
|---|---|---|---|---|---|---|
| Intel laptop, all = 22 | 20.7 | 32.1 | 42.6 | 53.3 | 60.2 | 61.3 |
| Intel workstation, all = 8 | 21.2 | 25.3 | 28.9 | 27.5 | 27.4 | 27.5 |
| Spark Arm CPU, all = 20 | 33.8 | 61.9 | 98.0 | 114.1 | 112.7 | 114.0 |

The GPU read sweeps, beside them: laptop RTX 3000 Ada 219 GB/s and Spark GB10 255 GB/s at 96 blocks; the workstation's GPU was not run.

What these runs show, and what they do not:

- **Every llama.cpp backend gave its own probabilities.** Observed: four runs, four values of `' Paris'`. The course's own NumPy rebuild gave 0.9260 on all four processors.
	- **CPU versus GPU matters**, observed: on the same Spark, the CPU build and the GPU build differ (0.9233 against 0.9272).
	- **The CPU matters too**, observed: two Linux gcc CPU builds, x86 and Arm, differ (0.9260 against 0.9233). So "the CPU path" is not one path.
	- **Windows and MSVC are not isolated:** the two x86 runs differ (0.9242 against 0.9260), but they differ in CPU as well as in compiler and operating system.
	- **Why, inferred and not tested here:** llama.cpp's CPU backend picks its matrix kernels by the instruction set of the CPU it was built on, and each kernel adds up its products in a different order and precision, so each rounds differently.
- **Decode runs at memory speed on every CPU.** Observed: decode tokens/s times the 7.16 GB model file comes to 60, 25 and 79 GB/s, against measured reads of 61.3, 27.5 and 114.0 GB/s.
- **The prefill ÷ decode ratio does not test MSVC.** It divides compute speed by memory speed, and the workstation's ratio (11x) is high mainly because its memory is slow (3.5 tokens/s decode), not because its prefill is fast. The Spark's Arm CPU, built with gcc on Linux, managed only 5x.
	- **Observed:** the laptop prefilled slowest of the three CPUs, 25 tokens/s against the workstation's 40 on half as many cores.
	- **Inferred:** that fits the MSVC build being slower, and fits the laptop's power limits and mixed fast and slow cores just as well. A gcc or clang build on the same laptop, under WSL, would separate them.

On Windows, `start_server.sh` ran in Git Bash with the native `llama-server.exe`: Git Bash converted the `/c/...` model path, and both `kill "$(cat /tmp/llm-bytes-up-server.pid)"` and the timeout stopped the native server.

## License

MIT — see [LICENSE](LICENSE). The model the scripts read is not part of this repository and carries its own license.
