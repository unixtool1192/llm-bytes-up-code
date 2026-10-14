"""`llmbytes exercise ID`: start or re-run one lesson's code exercise.

The exercise's tests are `exercises/test_<id>_<name>.py`, and they grade the functions in
`exercises/my_<name>.py`. If that file does not exist yet, this copies `exercises/stubs/<name>.py`
there, with the function signatures to fill in. Then it runs the tests. Your file is never
overwritten.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXERCISES = REPO / "exercises"


def available() -> dict[str, tuple[str, Path]]:
    """Exercise ID -> (name, test file), read from the test files' names: test_4b_softmax.py."""
    found = {}
    for test in sorted(EXERCISES.glob("test_*_*.py")):
        _, ident, name = test.stem.split("_", 2)
        found[ident] = (name, test)
    return found


def run(ident: str, pytest_args: list[str] | None = None) -> int:
    exercises = available()
    ident = ident.lower()
    if ident not in exercises:
        print(f"no exercise {ident!r}; the exercises are {', '.join(exercises)}", file=sys.stderr)
        return 2
    name, test = exercises[ident]
    mine = EXERCISES / f"my_{name}.py"
    if os.environ.get("LLMBYTES_SOLUTIONS", "") not in ("", "0"):
        print(f"LLMBYTES_SOLUTIONS is set: grading exercises/solutions/{name}.py, not yours\n")
    elif not mine.exists():
        shutil.copyfile(EXERCISES / "stubs" / f"{name}.py", mine)
        print(f"created exercises/{mine.name} from the stub: write your code there, then run "
              f"`uv run llmbytes exercise {ident}` again. Its tests fail until you do.\n")
    else:
        print(f"grading exercises/{mine.name}\n")
    sys.stdout.flush()  # print before pytest's own output, which comes from another process
    cmd = [sys.executable, "-m", "pytest", str(test.relative_to(REPO)), *(pytest_args or [])]
    return subprocess.call(cmd, cwd=REPO)
