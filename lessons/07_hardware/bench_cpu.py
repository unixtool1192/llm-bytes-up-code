"""Lesson 7 - The hardware fit, on a CPU.

The same measurements as bench.cu, for a machine without an NVIDIA GPU:
  1. what the machine says about itself
  2. how fast it can read memory
  3. how long multiplying one LLM weight matrix by N token vectors takes, for N = 1 .. 4096
     (the shape of blk.N.ffn_gate in the course's model: 11008 x 4096, here in FP32)
  4. peak matrix-multiply throughput (FP32; a CPU has no tensor cores)

The output has the same layout as bench.cu's, so the lesson's roofline chart reads either.

    uv run python lessons/07_hardware/bench_cpu.py | tee my_bench.txt
    uv run python lessons/07_hardware/bench_cpu.py --read-only --threads 1,2,4,all

Two things in here are less obvious than they look:

- The memory read uses max(), not sum(). NumPy's float32 sum does enough arithmetic per byte
  that it measures the CPU, not the memory: about 11 GB/s on one core of a machine whose memory
  reads at 75. A benchmark of memory has to do almost nothing with each byte it reads.
- NumPy runs a reduction such as max() on one thread. To use every core, the buffer is cut into
  many small slices and handed to a pool of threads; NumPy releases Python's global lock while
  it works, so the threads really do run at the same time. Many slices rather than one per
  thread lets fast cores take more of the work on CPUs that mix fast and slow cores.
"""

import argparse
import os
import platform
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import psutil
from threadpoolctl import threadpool_info

ALL = os.cpu_count() or 1
ROWS, COLS, MAX_N = 11008, 4096, 4096


def timeit(fn, warm=2, min_reps=5, max_reps=50, min_total=0.5):
    """Median and best wall time of fn(), in seconds, after warm-up runs."""
    for _ in range(warm):
        fn()
    times = []
    start = time.perf_counter()
    while len(times) < max_reps and (len(times) < min_reps or time.perf_counter() - start < min_total):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), min(times)


def cpu_name():
    if platform.system() == "Windows":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    if os.path.exists("/proc/cpuinfo"):
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    # Arm Linux has no "model name" in /proc/cpuinfo; lscpu decodes the core types instead,
    # one "Model name" line per kind of core.
    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=5).stdout
        names = [l.split(":", 1)[1].strip() for l in out.splitlines() if l.startswith("Model name")]
        if names:
            return " + ".join(dict.fromkeys(names))
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or platform.machine()


def thread_counts(spec):
    """'1,2,4,all' -> [1, 2, 4, <cpu count>]."""
    counts = []
    for part in spec.split(","):
        part = part.strip().lower()
        counts.append(ALL if part == "all" else int(part))
    if any(n < 1 for n in counts):
        raise SystemExit("--threads: every count must be at least 1")
    return counts


def section_device():
    vm = psutil.virtual_memory()
    print("== 1. the machine ==")
    print(f"  cpu:                           {cpu_name()}")
    print(f"  physical cores:                {psutil.cpu_count(logical=False)}")
    print(f"  logical processors:            {ALL}")
    print(f"  RAM total:                     {vm.total / 1e9:.1f} GB")
    print(f"  RAM free right now:            {vm.available / 1e9:.1f} GB")
    print(f"  numpy:                         {np.__version__}")
    for b in (i for i in threadpool_info() if i.get("user_api") == "blas"):
        print(f"  BLAS:                          {b['internal_api']} {b['version']} "
              f"({b.get('architecture', '?')} kernels, {b['num_threads']} threads)")
    return vm.available


def section_bandwidth(available, counts):
    print("\n== 2. read bandwidth (NumPy max over a buffer far bigger than any cache) ==")
    nbytes = min(1 << 30, int(available * 0.25)) // 4 * 4
    buf = np.full(nbytes // 4, 1.0, dtype=np.float32)       # writing it first touches every page
    chunks = np.array_split(buf, 128)
    gb = buf.nbytes / 1e9
    for n in counts:
        with ThreadPoolExecutor(n) as pool:
            t, _ = timeit(lambda: list(pool.map(np.max, chunks)), warm=2, min_reps=10)
        label = f"{n} thread{'s' if n > 1 else ''}"
        print(f"  {label:<11} read {gb:.2f} GB in {t * 1e3:7.2f} ms  ->  {gb / t:.1f} GB/s")


def section_matrix():
    # C (rows x N) = W (rows x cols) . X (cols x N). FP32: 2 FLOPs per 4-byte weight per token.
    print(f"\n== 3. weight matrix ({ROWS} x {COLS}, FP32) times N token vectors ==")
    rng = np.random.default_rng(0)
    W = (rng.random((ROWS, COLS), dtype=np.float32) - 0.5) * 0.1
    print(f"  {'N':>6} {'time (ms)':>10} {'TFLOP/s':>12} {'weights GB/s':>14} {'ms per token':>12}")
    n = 1
    while n <= MAX_N:
        X = rng.random((COLS, n), dtype=np.float32) - 0.5
        C = np.empty((ROWS, n), dtype=np.float32)              # allocated once, outside the timing
        t, _ = timeit(lambda: np.matmul(W, X, out=C), warm=2, min_reps=5 if n < 1024 else 3)
        flops = 2.0 * ROWS * COLS * n
        print(f"  {n:6d} {t * 1e3:10.3f} {flops / t / 1e12:12.3f} {W.nbytes / t / 1e9:14.1f} {t * 1e3 / n:12.5f}")
        n *= 2
    print("  (N = 1 runs BLAS's matrix-vector routine and N >= 2 its matrix-matrix one, so the step")
    print("   from 1 to 2 is a change of code path, not of hardware. Compare N = 2 onwards.)")


def section_peak():
    S = 4096
    print(f"\n== 4. big square matrix multiply ({S} x {S} x {S}, FP32) ==")
    rng = np.random.default_rng(1)
    A = rng.random((S, S), dtype=np.float32)
    B = rng.random((S, S), dtype=np.float32)
    C = np.empty((S, S), dtype=np.float32)
    t, best = timeit(lambda: np.matmul(A, B, out=C), warm=1, min_reps=5, min_total=1.0)
    big = 2.0 * S ** 3
    print(f"  FP32, {ALL} threads:   {t * 1e3:8.1f} ms  ->  {big / t / 1e12:6.3f} TFLOP/s  (best {big / best / 1e12:.3f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--read-only", action="store_true", help="only sections 1 and 2")
    ap.add_argument("--threads", default=f"1,{ALL}", metavar="LIST",
                    help="thread counts for the read test, e.g. 1,2,4,all (default: 1,all)")
    args = ap.parse_args()
    counts = thread_counts(args.threads)

    available = section_device()
    section_bandwidth(available, counts)
    if args.read_only:
        return
    section_matrix()
    section_peak()


if __name__ == "__main__":
    main()
