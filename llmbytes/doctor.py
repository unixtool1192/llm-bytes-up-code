"""`llmbytes doctor` — what this machine can run, and what to fix first.

It looks for the four things the lessons need — the model file, a llama-server, the CUDA
toolkit, and enough memory — measures memory bandwidth (lesson 4's lab predicts generation speed
from it), and prints which lessons are ready.

The probes (running nvidia-smi, hashing the model, timing a memory read) are kept apart from
the decisions (which tier, which lessons are ready), so the decisions can be tested without the
hardware: `gather()` collects `Facts`; everything after it is plain functions of `Facts`.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import psutil

from . import hardware

# The file the course's runs used. Only this exact file reproduces the lessons' probabilities.
MODEL_NAME = "deepseek-coder-6.7b-instruct.Q8_0.gguf"
MODEL_SIZE = 7_163_879_648
MODEL_SHA256 = "02cd6ce7ccec670cf6d3dd147932f13e584f9e964d5a3297a74b401b658471ae"
MODEL_URL = "https://huggingface.co/TheBloke/deepseek-coder-6.7B-instruct-GGUF"
# Bytes read per generated token: essentially the whole file (lesson 1).
BYTES_PER_TOKEN = 7.0e9

REPO = Path(__file__).resolve().parent.parent
BENCH_DIR = REPO / "lessons" / "07_hardware"
CACHE = Path.home() / ".cache" / "llm-bytes-up" / "model-hash.json"


@dataclass
class Facts:
    os: str = ""
    cpu: str = ""
    ram_total_gb: float = 0.0
    ram_free_gb: float = 0.0
    apple_silicon: bool = False
    gpu_name: str | None = None
    gpu_mem_gb: float | None = None
    nvcc: str | None = None              # CUDA release, e.g. "13.0"
    model_path: str | None = None
    model_size: int | None = None
    model_sha_ok: bool | None = None     # None: not hashed (missing, or --quick)
    server_bin: str | None = None
    server_version: str | None = None
    server_url: str = "http://127.0.0.1:8099"
    server_up: bool = False
    cpu_bandwidth: float | None = None   # GB/s, measured
    gpu_bandwidth: float | None = None   # GB/s, measured by bench --read-only, if built
    notes: list[str] = field(default_factory=list)


# ---- parsing (pure) ---------------------------------------------------------------------

def parse_nvidia_smi(out: str) -> tuple[str, float] | None:
    """'NVIDIA GB10, 122572' or 'NVIDIA RTX ..., 8188 MiB' -> (name, GB)."""
    line = out.strip().splitlines()[0] if out.strip() else ""
    m = re.match(r"(.+?),\s*([\d.]+|\[N/A\])", line)
    if not m:
        return None
    mem = float(m.group(2)) * 1024**2 / 1e9 if m.group(2)[0].isdigit() else None
    return m.group(1).strip(), mem


def parse_nvcc(out: str) -> str | None:
    m = re.search(r"release ([\d.]+)", out)
    return m.group(1) if m else None


def parse_llama_version(out: str) -> str | None:
    """Both formats llama.cpp has printed: 'version: 10630 (d222767c7)' and, in newer builds,
    'version: 0.3.0-dev (build 10630, commit d222767c7)'."""
    m = re.search(r"build (\d+), commit ([0-9a-f]+)", out) or re.search(r"version:\s*(\d+)\s*\(([0-9a-f]+)\)", out)
    return f"build {m.group(1)} ({m.group(2)})" if m else None


def parse_bench_bandwidth(out: str) -> float | None:
    rates = re.findall(r"->\s*([\d.]+)\s*GB/s", out)
    return max(float(r) for r in rates) if rates else None


# ---- decisions (pure) -------------------------------------------------------------------

def tier(f: Facts) -> str:
    if f.gpu_name:
        return "NVIDIA GPU"
    if f.apple_silicon:
        return "Apple silicon"
    return "CPU"


def decode_ceiling(f: Facts) -> tuple[float, str] | None:
    """Lesson 4's prediction: bandwidth / bytes per token, from whichever memory the model runs in.

    - A GPU that holds the model runs it at GPU bandwidth. An integrated GPU, such as the
      Spark's, reports no dedicated memory (None) and shares the machine's, so it counts.
    - A GPU too small for the 7 GB model plus its cache runs partly from CPU memory, so the
      CPU number is the honest one there.
    - A GPU that holds the model but has not been measured gets no prediction rather than a
      CPU number that would be wrong by 2x or more; the report says to build bench.
    """
    gpu_holds_model = f.gpu_name is not None and (f.gpu_mem_gb is None or f.gpu_mem_gb >= 9)
    if gpu_holds_model:
        return (f.gpu_bandwidth * 1e9 / BYTES_PER_TOKEN, "GPU") if f.gpu_bandwidth else None
    if f.cpu_bandwidth:
        return f.cpu_bandwidth * 1e9 / BYTES_PER_TOKEN, "CPU"
    return None


def lesson_status(f: Facts) -> list[tuple[str, bool, str]]:
    """(lesson, ready, what is missing or how it will run) for lessons 1-8."""
    have_model = f.model_path is not None
    exact = " (a different file: numbers will not match the lessons exactly)" if f.model_sha_ok is False else ""
    model_msg = "needs the model file: pass --model PATH or set MODEL"
    server_msg = "needs llama-server: set LLAMA_SERVER, then scripts/start_server.sh"
    rows = []
    for n in ("1", "2", "3"):
        rows.append((n, have_model, ("ready" + exact) if have_model else model_msg))
    for n in ("4", "5"):
        ok = have_model and f.server_bin is not None
        rows.append((n, ok, ("ready" + exact) if ok else (model_msg if not have_model else server_msg)))
    if not have_model:
        rows.append(("6", False, model_msg))
    elif f.ram_total_gb < 12:
        rows.append(("6", False, f"needs about 9 GB free memory; this machine has {f.ram_total_gb:.0f} GB in all"))
    else:
        rows.append(("6", True, "ready" + exact))
    if f.gpu_name and f.nvcc:
        rows.append(("7", True, "ready: bench.cu (make && ./bench)"))
    elif f.gpu_name:
        rows.append(("7", True, "ready on the CPU (bench_cpu.py); install the CUDA toolkit to run bench.cu"))
    else:
        rows.append(("7", True, "ready: bench_cpu.py (no NVIDIA GPU)"))
    ok8 = have_model and f.server_bin is not None
    rows.append(("8", ok8, "ready: start the server with LESSON_PARALLEL=16 LESSON_CTX=16384" if ok8
                 else (model_msg if not have_model else server_msg)))
    return rows


# ---- probes (touch the machine) -----------------------------------------------------------

def _run(cmd: list[str], env: dict | None = None, timeout: int = 60) -> str:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        return p.stdout + p.stderr
    except (OSError, subprocess.SubprocessError):
        return ""


def sha256_cached(path: Path) -> str:
    """Hash a 7 GB file once; reuse the answer while its size and mtime are unchanged."""
    st = path.stat()
    key = f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}"
    try:
        cache = json.loads(CACHE.read_text())
    except (OSError, ValueError):
        cache = {}
    if key in cache:
        return cache[key]
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(16 << 20), b""):
            h.update(block)
    digest = h.hexdigest()
    try:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        cache[key] = digest
        CACHE.write_text(json.dumps(cache, indent=1))
    except OSError:
        pass
    return digest


def _find_bench() -> Path | None:
    for name in ("bench", "bench.exe"):
        p = BENCH_DIR / name
        if p.is_file():
            return p
    return None


def gather(model: str | None, server_bin: str | None, server_url: str, quick: bool) -> Facts:
    f = Facts(server_url=server_url)
    f.os = f"{platform.system()} {platform.release()} ({platform.machine()})"
    f.cpu = hardware.cpu_name()
    vm = psutil.virtual_memory()
    f.ram_total_gb, f.ram_free_gb = vm.total / 1e9, vm.available / 1e9
    f.apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"

    if shutil.which("nvidia-smi"):
        parsed = parse_nvidia_smi(_run(["nvidia-smi", "--query-gpu=name,memory.total",
                                        "--format=csv,noheader,nounits"]))
        if parsed:
            f.gpu_name, f.gpu_mem_gb = parsed
    if shutil.which("nvcc"):
        f.nvcc = parse_nvcc(_run(["nvcc", "--version"]))

    model = model or os.environ.get("MODEL")
    if model:
        p = Path(model)
        if p.is_file():
            f.model_path, f.model_size = str(p), p.stat().st_size
            if f.model_size != MODEL_SIZE:
                f.model_sha_ok = False
            elif not quick:
                f.model_sha_ok = sha256_cached(p) == MODEL_SHA256
        else:
            f.notes.append(f"no file at {model}")

    server_bin = server_bin or os.environ.get("LLAMA_SERVER") or shutil.which("llama-server")
    if server_bin and Path(server_bin).is_file():
        f.server_bin = server_bin
        env = dict(os.environ)
        # Same as start_server.sh: a build that keeps its libraries beside the binary needs them.
        env["LD_LIBRARY_PATH"] = str(Path(server_bin).parent) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
        f.server_version = parse_llama_version(_run([server_bin, "--version"], env=env))
    try:
        with urllib.request.urlopen(server_url.rstrip("/") + "/health", timeout=2) as r:
            f.server_up = b'"ok"' in r.read()
    except OSError:
        f.server_up = False

    buf = hardware.read_buffer(vm.available)
    f.cpu_bandwidth, _ = hardware.read_bandwidth(buf, hardware.ALL_THREADS)
    del buf
    bench = _find_bench()
    if f.gpu_name and bench:
        f.gpu_bandwidth = parse_bench_bandwidth(_run([str(bench), "--read-only"], timeout=120))
    return f


# ---- report ---------------------------------------------------------------------------------

def report(f: Facts, lesson: str | None = None) -> list[str]:
    ok, no, info = "✓", "✗", "ℹ"
    out = [f"machine     {f.cpu}, {f.ram_total_gb:.0f} GB memory ({f.ram_free_gb:.0f} GB free), {f.os}"]
    if f.gpu_name:
        mem = f", {f.gpu_mem_gb:.1f} GB" if f.gpu_mem_gb else ""
        out.append(f"gpu         {f.gpu_name}{mem}; CUDA toolkit {f.nvcc or 'not found (needed only for bench.cu)'}")
    elif f.apple_silicon:
        out.append("gpu         Apple silicon: llama.cpp runs on Metal; lesson 7 runs bench_cpu.py")
    else:
        out.append("gpu         no NVIDIA GPU found: every lesson runs on the CPU")

    if f.model_path is None:
        out.append(f"{no} model     not found. Download {MODEL_NAME} from {MODEL_URL}, then pass --model PATH")
    elif f.model_sha_ok is True:
        out.append(f"{ok} model     {f.model_path}  (the course's exact file: SHA-256 matches)")
    elif f.model_sha_ok is False:
        out.append(f"{info} model     {f.model_path} is a different file from the course's "
                   f"({f.model_size:,} bytes). Lessons run, but probabilities will not match exactly")
    else:
        out.append(f"{info} model     {f.model_path}  (size matches; not hashed because of --quick)")

    if f.server_bin:
        out.append(f"{ok} server    {f.server_bin}  {f.server_version or '(version not readable)'}"
                   f"{'; running at ' + f.server_url if f.server_up else ''}")
    else:
        out.append(f"{no} server    llama-server not found: build llama.cpp, then set LLAMA_SERVER=/path/to/llama-server")

    bw = f"bandwidth   CPU memory read {f.cpu_bandwidth:.0f} GB/s (measured)"
    if f.gpu_bandwidth:
        bw += f"; GPU {f.gpu_bandwidth:.0f} GB/s (bench --read-only)"
    elif f.gpu_name:
        bw += "; GPU not measured — build lessons/07_hardware/bench to measure it"
    out.append(bw)
    ceiling = decode_ceiling(f)
    if ceiling:
        out.append(f"            lesson 4 prediction: ceiling about {ceiling[0]:.0f} tokens/s when the model runs on the {ceiling[1]}")
    if f.gpu_name and f.gpu_mem_gb is not None and f.gpu_mem_gb < 9:
        out.append(f"            the 7 GB model plus about 2 GB of cache does not fit in {f.gpu_mem_gb:.1f} GB of GPU memory;"
                   " start the server with LESSON_GPU_LAYERS=0, or a number below 32 to split it")

    out.append(f"tier        {tier(f)}")
    out.append("")
    for n, ready, msg in lesson_status(f):
        if lesson and n != str(int(lesson)):
            continue
        out.append(f"{ok if ready else no} lesson {n}   {msg}")
    out.extend(f"{info} {n}" for n in f.notes)
    return out
