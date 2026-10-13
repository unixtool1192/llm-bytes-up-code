"""`llmbytes doctor`'s decisions, tested without the hardware.

`gather()` probes the machine; everything tested here is a plain function of the `Facts` it
returns, so each case below is a machine described in a few fields.
"""

from __future__ import annotations

from llmbytes import doctor
from llmbytes.doctor import Facts


def _status(facts):
    return {n: (ready, msg) for n, ready, msg in doctor.lesson_status(facts)}


def test_parse_nvidia_smi():
    assert doctor.parse_nvidia_smi("NVIDIA RTX 3000 Ada Generation Laptop GPU, 8188\n") == (
        "NVIDIA RTX 3000 Ada Generation Laptop GPU", 8188 * 1024**2 / 1e9)
    # An integrated GPU can report no dedicated memory.
    assert doctor.parse_nvidia_smi("NVIDIA GB10, [N/A]\n") == ("NVIDIA GB10", None)
    assert doctor.parse_nvidia_smi("") is None


def test_parse_versions():
    assert doctor.parse_nvcc("Cuda compilation tools, release 13.0, V13.0.88") == "13.0"
    assert doctor.parse_llama_version("version: 10630 (d222767c7)\nbuilt with ...") == "build 10630 (d222767c7)"
    # The format the course's own build prints (seen on the Spark, 2026-10-13).
    assert doctor.parse_llama_version(
        "version: 0.3.0-dev (build 10630, commit d222767c7)\nbuilt with GNU 13.3.0") == "build 10630 (d222767c7)"
    assert doctor.parse_llama_version("error while loading shared libraries") is None


def test_parse_bench_bandwidth_takes_the_best_line():
    out = "  1 blocks  read 2.15 GB in 91 ms  ->  24 GB/s\n  16 blocks  read 2.15 GB in 8.9 ms  ->  242 GB/s\n"
    assert doctor.parse_bench_bandwidth(out) == 242


def test_bare_laptop_can_still_do_lesson_7():
    s = _status(Facts(ram_total_gb=16, cpu_bandwidth=60))
    assert s["7"][0] and "bench_cpu.py" in s["7"][1]
    assert not s["1"][0] and "--model" in s["1"][1]
    assert doctor.tier(Facts()) == "CPU"


def test_model_without_server_readies_file_lessons_only():
    s = _status(Facts(ram_total_gb=32, model_path="m.gguf", model_sha_ok=True))
    assert all(s[n][0] for n in ("1", "2", "3", "6"))
    assert not s["4"][0] and "LLAMA_SERVER" in s["4"][1]


def test_small_machine_cannot_rebuild_the_model():
    s = _status(Facts(ram_total_gb=8, model_path="m.gguf", model_sha_ok=True))
    assert not s["6"][0] and "9 GB" in s["6"][1]


def test_different_model_file_is_flagged_not_blocked():
    s = _status(Facts(ram_total_gb=32, model_path="m.gguf", model_sha_ok=False, server_bin="llama-server"))
    assert s["4"][0] and "not match" in s["4"][1]


def test_gpu_without_toolkit_falls_back_to_cpu_benchmark():
    s = _status(Facts(gpu_name="NVIDIA RTX", nvcc=None))
    assert s["7"][0] and "CUDA toolkit" in s["7"][1]
    assert _status(Facts(gpu_name="NVIDIA RTX", nvcc="13.0"))["7"][1].startswith("ready: bench.cu")


def test_decode_ceiling_uses_the_memory_the_model_runs_in():
    # Integrated (no dedicated-memory figure): the GPU reads the shared memory at GPU speed.
    # Measured on the Spark: 259 GB/s / 7 GB = 37 tokens/s ceiling, and llama.cpp decoded at 33.7.
    spark = Facts(gpu_name="NVIDIA GB10", gpu_mem_gb=None, gpu_bandwidth=259, cpu_bandwidth=115)
    rate, where = doctor.decode_ceiling(spark)
    assert where == "GPU" and 33.7 < rate < 40
    # Same machine before bench is built: no prediction, rather than the CPU's 16 tokens/s.
    assert doctor.decode_ceiling(Facts(gpu_name="NVIDIA GB10", gpu_mem_gb=None, cpu_bandwidth=115)) is None
    roomy = Facts(gpu_name="H100", gpu_mem_gb=80, gpu_bandwidth=3000, cpu_bandwidth=100)
    rate, where = doctor.decode_ceiling(roomy)
    assert where == "GPU" and round(rate) == round(3000e9 / doctor.BYTES_PER_TOKEN)
    small = Facts(gpu_name="RTX 3000", gpu_mem_gb=8.6, gpu_bandwidth=244, cpu_bandwidth=69)
    assert doctor.decode_ceiling(small)[1] == "CPU"     # 8.6 GB cannot hold the 7 GB model plus its cache
    assert doctor.decode_ceiling(Facts()) is None


def test_report_names_the_download_when_the_model_is_missing():
    text = "\n".join(doctor.report(Facts(cpu="x", cpu_bandwidth=50)))
    assert doctor.MODEL_URL in text and "--model" in text


def test_report_lesson_filter():
    lines = doctor.report(Facts(cpu="x", cpu_bandwidth=50), lesson="7")
    assert sum("lesson " in l for l in lines if l.startswith(("✓", "✗"))) == 1
