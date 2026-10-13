"""`llmbytes check NN FILE` — compare a lesson run with the course's recorded run.

A lesson's output mixes two kinds of line, and they are checked differently:

- **Lines with no time or speed in them** — probabilities, token IDs, tensor shapes, cosine
  similarities — depend only on the model file and the arithmetic, so they must match the
  reference on any hardware. Numbers may differ in their last printed digit (a different BLAS
  library rounds slightly differently), and column spacing may differ; nothing else may.
- **Measurements** — anything in ms, s, tokens/s, GB/s or FLOP/s — depend on the machine, so
  they are never compared one to one. Where a lesson's point is a *shape* (prefill is far faster
  than decode; time stays flat while memory-bound), that shape is checked instead.

Lessons 7 and 8 are all measurement, so they get shape checks only. Lesson 7's checks read
nothing from a reference — they judge a run against its own bandwidth and peak — so the same
check works for bench.cu and bench_cpu.py on any machine.

The result is a list of `Finding`s; `ok` is True, False, or None for information only. Any
False makes the command exit 1.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REFERENCE = REPO / "outputs" / "2026-10-12"

# A number with an optional decimal part and exponent.
_NUM = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?")
# A line holding a measurement: a number followed by a unit of time or speed, or a timing label.
_MEASURE = re.compile(
    r"\d\s*(?:ms|s|seconds|tokens/s|tok/s|GB/s|[GT]FLOP/s)\b|\bwall\b|max RSS|x the time|free right now",
    re.IGNORECASE,
)
# Lines only present when the lesson ran with --server, to cross-check against llama.cpp.
_SERVER_ONLY = re.compile(r"llama\.cpp")


@dataclass
class Finding:
    ok: bool | None
    text: str


def reference_for(lesson: str, ref_dir: Path = REFERENCE) -> Path:
    """The reference file for a lesson number like '04' or '4'."""
    num = f"{int(lesson):02d}"
    candidates = sorted(ref_dir.glob(f"{num}_*.txt"))
    if not candidates:
        raise SystemExit(f"no reference output for lesson {num} in {ref_dir}")
    return candidates[0]


def is_measurement(line: str) -> bool:
    return bool(_MEASURE.search(line))


def is_note(line: str) -> bool:
    """A line that is only a parenthetical remark is prose, not a result. Lesson 5 gained one
    ('by hand uses only the top 200 tokens...') seconds after its reference was recorded, so
    comparing notes would fail every correct run."""
    s = line.strip()
    return s.startswith("(") and s.endswith(")")


def _numbers_match(a: str, b: str) -> bool:
    fa, fb = float(a.replace(",", "")), float(b.replace(",", ""))
    if "e" in a.lower() or "e" in b.lower():
        # Tiny residuals such as a largest logit difference of 1.6e-05 depend on summation
        # order; the same order of magnitude is the meaningful match.
        return fa == fb or (fa != 0 and fb != 0 and 0.1 <= abs(fa / fb) <= 10)
    decimals = max(len(x.split(".")[1]) if "." in x else 0 for x in (a, b))
    return abs(fa - fb) <= 2 * 10 ** -decimals + 1e-12


def lines_match(mine: str, ref: str) -> bool:
    """Equal apart from spacing and the last digit of each number."""
    split_m, split_r = _NUM.split(mine), _NUM.split(ref)
    nums_m, nums_r = _NUM.findall(mine), _NUM.findall(ref)
    if len(nums_m) != len(nums_r):
        return False
    norm = lambda parts: [" ".join(p.split()) for p in parts]
    if norm(split_m) != norm(split_r):
        return False
    return all(_numbers_match(a, b) for a, b in zip(nums_m, nums_r))


_SERVER_PART = re.compile(r"\s*(?:vs\s+)?llama\.cpp(?:\s+[-+]?[\d.]+)?")


def without_server(ref: list[str]) -> tuple[list[str], int]:
    """The reference as a run without --server would print it.

    The course's runs used --server, which adds llama.cpp's own answer beside the rebuilt one:
    whole lines ('      llama.cpp MATCH'), a value on a line ('rebuild 0.9260   llama.cpp 0.9272'),
    and words in a heading ('(rebuilt tokenizer vs llama.cpp)'). Whole lines are dropped; the
    rest keep their own numbers, which are still checked. Returns (lines, cross-checks removed).
    """
    out, removed = [], 0
    for line in ref:
        if not _SERVER_ONLY.search(line):
            out.append(line)
        elif line.strip().startswith("llama.cpp"):
            removed += 1
        else:
            out.append(_SERVER_PART.sub("", line))
            removed += 1
    return out, removed


def compare_exact(mine: list[str], ref: list[str], max_shown: int = 5) -> list[Finding]:
    """Compare the hardware-independent lines of two runs."""
    keep = lambda l: l.strip() and not is_measurement(l) and not is_note(l)
    m = [l for l in mine if keep(l)]
    r = [l for l in ref if keep(l)]
    findings: list[Finding] = []
    bad: list[str] = []
    skipped_server = 0
    if not any(_SERVER_ONLY.search(l) for l in m):
        r, skipped_server = without_server(r)
    matched = 0
    sm = difflib.SequenceMatcher(a=r, b=m, autojunk=False)
    for op, r0, r1, m0, m1 in sm.get_opcodes():
        if op == "equal":
            matched += r1 - r0
        elif op == "replace" and r1 - r0 == m1 - m0:
            for rl, ml in zip(r[r0:r1], m[m0:m1]):
                if lines_match(ml, rl):
                    matched += 1
                else:
                    bad.append(f"expected: {rl.strip()}\n       got: {ml.strip()}")
        else:
            for rl in r[r0:r1]:
                bad.append(f"missing:  {rl.strip()}")
            for ml in m[m0:m1]:
                bad.append(f"extra:    {ml.strip()}")
    if bad:
        findings.append(Finding(False, f"{len(bad)} line(s) that do not depend on hardware differ from the reference:"))
        findings.extend(Finding(False, "  " + b) for b in bad[:max_shown])
        if len(bad) > max_shown:
            findings.append(Finding(False, f"  ... and {len(bad) - max_shown} more"))
    else:
        findings.append(Finding(True, f"{matched} lines that do not depend on hardware match the reference"))
    if skipped_server:
        findings.append(Finding(None, f"{skipped_server} llama.cpp cross-check line(s) skipped: run the lesson with --server to include them"))
    return findings


# ---- shape checks -------------------------------------------------------------------------

def _first_float(pattern: str, text: str) -> list[float]:
    return [float(x.replace(",", "")) for x in re.findall(pattern, text)]


def shape_03(text: str, _ref: str) -> list[Finding]:
    m = re.search(r"\(([\d.]+)x the time for 256x the work\)", text)
    if not m:
        return [Finding(False, "no matvec-vs-matmul timing line found")]
    x = float(m.group(1))
    return [Finding(x < 64, f"256 tokens took {x:.1f}x the time of 1 token, for 256x the work "
                            f"(well under 256x means reading the matrix once for all of them paid off)")]


def shape_04(text: str, ref: str) -> list[Finding]:
    def rates(t):
        decode = _first_float(r"(\d[\d.]*) tokens/s\n", t)
        prefill = _first_float(r"= (\d+) tokens/s;", t)
        return (decode[0] if decode else None), (max(prefill) if prefill else None)
    d, p = rates(text)
    rd, rp = rates(ref)
    if d is None or p is None:
        return [Finding(False, "no decode or prefill timing found — did the run finish?")]
    out = [Finding(None, f"decode     yours {d:.1f} tokens/s   reference {rd:.1f}"),
           Finding(None, f"prefill    yours {p:,.0f} tokens/s  reference {rp:,.0f}")]
    ratio, rratio = p / d, rp / rd
    out.append(Finding(ratio > 3, f"prefill is {ratio:.0f}x faster per token than decode (reference {rratio:.0f}x) "
                                  "— the gap is smaller on a CPU, which has less arithmetic to spare"))
    return out


def _bench_rows(text: str) -> list[tuple[int, float, float, float]]:
    """(N, time ms, TFLOP/s, weights GB/s) from section 3 of bench.cu or bench_cpu.py."""
    sec = text.split("== 3.", 1)[-1].split("== 4.", 1)[0]
    rows = re.findall(r"^\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+[\d.]+\s*$", sec, re.M)
    return [(int(n), float(t), float(f), float(g)) for n, t, f, g in rows]


def shape_07(text: str, ref: str) -> list[Finding]:
    rows = {n: (t, f, g) for n, t, f, g in _bench_rows(text)}
    bws = _first_float(r"->\s*([\d.]+)\s*GB/s", text.split("== 3.", 1)[0])
    if not rows or not bws or 2 not in rows or 32 not in rows:
        return [Finding(False, "could not read sections 2 and 3 — pass the whole output of bench or bench_cpu.py")]
    bw = max(bws)
    peak = max(f for _, f, _ in rows.values())
    fp32 = "FP32" in text.split("== 3.", 1)[-1].split("\n", 1)[0]
    bytes_per_weight = 4 if fp32 else 2
    ridge = peak * 1000 / bw                       # FLOPs per byte where the roofline bends
    ridge_n = ridge * bytes_per_weight / 2         # tokens sharing each weight read, at the ridge
    knee = min((n for n, (_, f, _) in sorted(rows.items()) if f >= 0.8 * peak), default=None)
    out = [Finding(None, f"bandwidth  {bw:.0f} GB/s (section 2)"),
           Finding(None, f"peak       {peak:.2f} TFLOP/s (section 3)"),
           Finding(None, f"ridge      {ridge:.0f} FLOPs per byte -> about N = {ridge_n:.0f} tokens for "
                         f"{'FP32' if fp32 else 'BF16'} weights")]
    flat = rows[32][0] / rows[2][0]
    out.append(Finding(flat < 4, f"memory-bound region: 16x the tokens (N = 2 -> 32) took {flat:.1f}x the time"))
    big = sorted(n for n in rows if n >= 1024)
    if len(big) >= 2:
        a, b = big[-2], big[-1]
        grow = rows[b][0] / rows[a][0]
        out.append(Finding(grow > 1.5, f"compute-bound region: doubling N ({a} -> {b}) took {grow:.1f}x the time"))
    if knee:
        out.append(Finding(None, f"crossover  throughput reaches 80% of peak at N = {knee}; the ridge predicts "
                                 f"about {ridge_n:.0f} (real kernels bend later than the ideal roofline)"))
    return out


def shape_08(text: str, _ref: str) -> list[Finding]:
    rows = re.findall(r"^\s*(\d+)\s+[\d.]+s\s+([\d.]+)\s+([\d.]+)\s+[\d.]+x\s*$", text, re.M)
    if len(rows) < 2:
        return [Finding(False, "no batching table found")]
    first, last = rows[0], rows[-1]
    gain = float(last[2]) / float(first[2])
    per_user = float(last[1]) / float(first[1])
    return [Finding(gain > 2, f"{last[0]} users together produced {gain:.1f}x the total tokens/s of 1 user"),
            Finding(None, f"while each user's own speed went to {per_user:.0%} of the single-user speed")]


SHAPES = {"03": shape_03, "04": shape_04, "07": shape_07, "08": shape_08}
ALL_MEASUREMENT = {"07", "08"}     # nothing in these depends only on the model


def run(lesson: str, text: str, ref_dir: Path = REFERENCE) -> tuple[Path, list[Finding]]:
    num = f"{int(lesson):02d}"
    if num not in {f"{n:02d}" for n in range(1, 9)}:
        raise SystemExit(f"no lesson {lesson}: the course has lessons 1-7 and the capstone, 8")
    ref_path = reference_for(num, ref_dir)
    ref = ref_path.read_text(encoding="utf-8")
    findings: list[Finding] = []
    if num not in ALL_MEASUREMENT:
        findings += compare_exact(text.splitlines(), ref.splitlines())
    if num in SHAPES:
        findings += SHAPES[num](text, ref)
    return ref_path, findings
