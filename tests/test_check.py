"""`llmbytes check` against the course's own recorded runs.

The recorded runs are the fixtures: a reference must pass against itself, a changed probability
must fail, a changed timing must not, and lesson 7's shape checks must pass on every machine the
course has been run on — a Spark GPU, a laptop GPU, and two CPUs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llmbytes import check

OUT = Path(__file__).resolve().parent.parent / "outputs"
REF = OUT / "2026-10-12"


def _failures(findings):
    return [f.text for f in findings if f.ok is False]


@pytest.mark.parametrize("lesson", ["01", "02", "03", "04", "05", "06", "07", "08"])
def test_every_reference_passes_against_itself(lesson):
    text = check.reference_for(lesson).read_text(encoding="utf-8")
    _, findings = check.run(lesson, text)
    assert not _failures(findings)


def test_changed_probability_fails():
    text = (REF / "04_next_token.txt").read_text(encoding="utf-8")
    bad = text.replace("p = 0.9272", "p = 0.9100", 1)
    assert bad != text
    _, findings = check.run("04", bad)
    fails = _failures(findings)
    assert fails and any("0.9100" in f for f in fails)


def test_last_digit_rounding_is_tolerated():
    text = (REF / "04_next_token.txt").read_text(encoding="utf-8")
    _, findings = check.run("04", text.replace("p = 0.9272", "p = 0.9273", 1))
    assert not _failures(findings)


def test_changed_token_fails_even_with_same_numbers():
    text = (REF / "04_next_token.txt").read_text(encoding="utf-8")
    _, findings = check.run("04", text.replace("' Paris'", "' Lyon'", 1))
    assert _failures(findings)


def test_timings_are_not_compared_one_to_one():
    """A slower machine is not a failure."""
    text = (REF / "04_next_token.txt").read_text(encoding="utf-8")
    slow = (text.replace("128 tokens at 33.7 tokens/s", "128 tokens at 9.1 tokens/s")
                .replace("= 2407 tokens/s;  then decode 29.3", "= 310 tokens/s;  then decode 8.0")
                .replace("= 2414 tokens/s;  then decode 29.2", "= 305 tokens/s;  then decode 8.1"))
    assert slow != text
    _, findings = check.run("04", slow)
    assert not _failures(findings)


def test_prefill_no_faster_than_decode_fails_the_shape():
    text = (REF / "04_next_token.txt").read_text(encoding="utf-8")
    flat = text.replace("= 2407 tokens/s;", "= 30 tokens/s;").replace("= 2414 tokens/s;", "= 31 tokens/s;")
    _, findings = check.run("04", flat)
    assert any("faster per token" in f for f in _failures(findings))


def _as_if_no_server(text: str) -> str:
    """What lessons 2 and 6 print without --server, derived from what they print with it:
    the heading loses ' vs llama.cpp', 'llama.cpp MATCH' lines vanish, and a trailing
    'llama.cpp 0.9272' column goes. (A real lesson 2 run without --server was compared on the
    Spark on 2026-10-13 and produced exactly these differences.)"""
    import re
    out = []
    for line in text.splitlines():
        if line.strip().startswith("llama.cpp"):
            continue
        out.append(re.sub(r"\s*(?:vs\s+)?llama\.cpp(?:\s+[-\d.]+)?", "", line))
    return "\n".join(out)


@pytest.mark.parametrize("lesson,name", [("02", "02_tokens_and_embeddings.txt"), ("06", "06_attention_rebuild.txt")])
def test_run_without_server_skips_crosschecks(lesson, name):
    _, findings = check.run(lesson, _as_if_no_server((REF / name).read_text(encoding="utf-8")))
    assert not _failures(findings), _failures(findings)
    assert any(f.ok is None and "--server" in f.text for f in findings)


def test_without_server_still_checks_the_rebuilds_own_numbers():
    """Dropping the llama.cpp column must not drop the rebuilt value beside it."""
    text = _as_if_no_server((REF / "06_attention_rebuild.txt").read_text(encoding="utf-8"))
    bad = text.replace("rebuild 0.9260", "rebuild 0.8000", 1)
    assert bad != text
    _, findings = check.run("06", bad)
    assert any("0.8000" in f for f in _failures(findings))


@pytest.mark.parametrize("run", sorted((OUT / "2026-10-13").glob("*/07_*.txt")), ids=lambda p: p.parent.name)
def test_lesson_7_shape_holds_on_every_recorded_machine(run):
    _, findings = check.run("07", run.read_text(encoding="utf-8"))
    assert not _failures(findings), _failures(findings)
    assert any(f.ok is True and "memory-bound" in f.text for f in findings)


def test_lesson_7_without_sections_fails_clearly():
    _, findings = check.run("07", "== 1. the device ==\n  name: x\n")
    assert any("sections 2 and 3" in f for f in _failures(findings))


def test_current_lesson_5_note_line_is_not_a_failure():
    """The lesson 5 script prints a caveat its 2026-10-12 reference predates (da410f7)."""
    text = (REF / "05_sampling.txt").read_text(encoding="utf-8")
    lines = text.splitlines()
    lines.insert(len(lines) // 2, "  (by hand uses only the top 200 tokens, so temperatures above 1 read slightly high)")
    _, findings = check.run("05", "\n".join(lines))
    assert not _failures(findings)


def test_note_beside_data_is_still_checked():
    assert not check.is_note("  256 tokens:  28.10 ms   821.6 GFLOP/s  (20.0x the time for 256x the work)")
    assert check.is_note("  (by hand uses only the top 200 tokens)")


def test_unknown_lesson_is_refused():
    with pytest.raises(SystemExit):
        check.run("12", "")


@pytest.mark.parametrize("line,expected", [
    ("forward pass: 4.7 s", True),
    ("  128 tokens at 33.7 tokens/s", True),
    ("  1 token:      1.41 ms   128.3 GB/s    64.2 GFLOP/s", True),
    ("wall 14.92s, max RSS 8618988KB", True),
    ("      2048 tokens -> 1.07 GB", False),
    ("  ' Paris'     id   8873  p = 0.9272   logprob = -0.0756", False),
    ("layers=32 heads=32 kv_heads=32 head_dim=128", False),
])
def test_measurement_lines_are_recognised(line, expected):
    assert check.is_measurement(line) is expected
