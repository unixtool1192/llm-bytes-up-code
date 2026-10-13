"""What the machine is, and how fast it reads memory.

Shared by lesson 7's CPU benchmark and by `llmbytes doctor`, so the bandwidth a learner is told
about in lesson 4's lab is measured exactly the way lesson 7 measures it.

The memory read uses max(), not sum(). NumPy's float32 sum does enough arithmetic per byte that
it measures the CPU rather than the memory. NumPy also runs a reduction on one thread, so the
buffer is cut into many slices and handed to a pool of threads; NumPy releases Python's global
lock while it works, so the threads really run at the same time. Many small slices, rather than
one per thread, let fast cores take more of the work on CPUs that mix fast and slow cores.
"""

from __future__ import annotations

import os
import platform
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ALL_THREADS = os.cpu_count() or 1


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


def cpu_name() -> str:
    if platform.system() == "Windows":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    if platform.system() == "Darwin":
        try:
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                 capture_output=True, text=True, timeout=5).stdout.strip()
            if out:
                return out
        except (OSError, subprocess.SubprocessError):
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


def read_buffer(available_bytes: int) -> np.ndarray:
    """A float32 buffer far bigger than any cache: 1 GiB, or a quarter of free memory if less.
    Writing it first touches every page, so the timing never includes the OS handing out memory."""
    nbytes = min(1 << 30, int(available_bytes * 0.25)) // 4 * 4
    return np.full(nbytes // 4, 1.0, dtype=np.float32)


def read_bandwidth(buf: np.ndarray, threads: int, slices: int = 128) -> tuple[float, float]:
    """Read `buf` with `threads` threads. Returns (GB/s, seconds per full read)."""
    chunks = np.array_split(buf, slices)
    with ThreadPoolExecutor(threads) as pool:
        t, _ = timeit(lambda: list(pool.map(np.max, chunks)), warm=2, min_reps=10)
    return buf.nbytes / 1e9 / t, t
