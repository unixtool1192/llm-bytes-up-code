"""pytest setup for the lesson exercises: say whose code is being graded."""

import sys
from pathlib import Path

# The test files import `_exercise` from this directory, however pytest was started.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _exercise import using_solutions  # noqa: E402


def pytest_report_header(config):
    if using_solutions():
        return "exercises: grading the course's solutions (LLMBYTES_SOLUTIONS is set)"
    return "exercises: grading your my_*.py files"
