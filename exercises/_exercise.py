"""Load the module an exercise test grades: yours, or the course's solution.

Each lesson's code exercise has a test file here, `test_<id>_<name>.py`, and you write the
functions it tests in `my_<name>.py`, beside it. `stubs/<name>.py` is the starting point;
`uv run llmbytes exercise <id>` copies it for you, then runs the tests.

Set LLMBYTES_SOLUTIONS=1 to grade `solutions/<name>.py` instead. That is how CI checks that
every test still passes against its solution.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent

# The exercises that read the model file need its path, as every lesson script does.
needs_model = pytest.mark.skipif(
    not os.environ.get("MODEL"),
    reason="this test reads the model file: set MODEL to its path, as in Lab Setup",
)


def using_solutions() -> bool:
    return os.environ.get("LLMBYTES_SOLUTIONS", "") not in ("", "0")


def learner(exercise: str, name: str):
    """Import `my_<name>.py`, or `solutions/<name>.py` under LLMBYTES_SOLUTIONS=1.

    Called at the top of a test file. When the file does not exist yet, the whole test file
    is skipped with a message saying how to create it, instead of failing with an ImportError.
    """
    if using_solutions():
        path, modname = HERE / "solutions" / f"{name}.py", f"solution_{name}"
        if not path.exists():
            # A missing solution is a repository bug, so fail loudly rather than skip in CI.
            raise FileNotFoundError(f"exercise {exercise}: no solution at {path}")
    else:
        path, modname = HERE / f"my_{name}.py", f"my_{name}"
    if not path.exists():
        pytest.skip(
            f"exercise {exercise}: exercises/{path.relative_to(HERE).as_posix()} does not exist "
            f"yet. Run `uv run llmbytes exercise {exercise}` to create it from "
            f"exercises/stubs/{name}.py, then fill it in.",
            allow_module_level=True,
        )
    spec = importlib.util.spec_from_file_location(modname, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
