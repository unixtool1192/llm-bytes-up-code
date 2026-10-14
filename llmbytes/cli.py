"""The `llmbytes` command: lab helpers for the course.

    uv run llmbytes doctor [--model PATH] [--lesson N] [--quick]
    uv run llmbytes check NN FILE        (FILE '-' reads standard input)
    uv run llmbytes exercise ID          (e.g. 4b: create exercises/my_*.py if missing, run its tests)
"""

from __future__ import annotations

import argparse
import sys

from . import check, doctor, exercise


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to a code page that cannot print ✓ and ✗.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(prog="llmbytes", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("doctor", help="what this machine can run, and what to fix first")
    d.add_argument("--model", help="path to the model file (default: $MODEL)")
    d.add_argument("--llama-server", help="path to llama-server (default: $LLAMA_SERVER, then PATH)")
    d.add_argument("--server", default="http://127.0.0.1:8099", help="where a running server would answer")
    d.add_argument("--lesson", help="show only this lesson's line")
    d.add_argument("--quick", action="store_true", help="skip hashing the model (it takes about a minute, once)")

    c = sub.add_parser("check", help="compare a lesson run with the course's recorded run")
    c.add_argument("lesson", help="lesson number, 1-8 (8 is the capstone)")
    c.add_argument("file", help="the lesson's saved output, or - for standard input")

    e = sub.add_parser("exercise", help="start or re-run a lesson's code exercise, e.g. 4b")
    e.add_argument("id", help="the exercise, as the lesson names it: 1b, 2b, 3b, 4b, 5b, 6b or 8b")

    args = ap.parse_args(argv)

    if args.cmd == "exercise":
        return exercise.run(args.id)

    if args.cmd == "doctor":
        facts = doctor.gather(args.model, args.llama_server, args.server, args.quick)
        print("\n".join(doctor.report(facts, args.lesson)))
        return 0

    text = sys.stdin.read() if args.file == "-" else open(args.file, encoding="utf-8").read()
    ref, findings = check.run(args.lesson, text)
    if f"{int(args.lesson):02d}" in check.ALL_MEASUREMENT:
        print(f"lesson {int(args.lesson)}: every number here depends on the machine, so your run is checked "
              f"for its shape, not against the course's values\n")
    else:
        print(f"lesson {int(args.lesson)}: your run vs {ref.relative_to(check.REPO).as_posix()}\n")
    marks = {True: "✓", False: "✗", None: "ℹ"}
    for f in findings:
        print(f"{marks[f.ok]} {f.text}")
    return 1 if any(f.ok is False for f in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
