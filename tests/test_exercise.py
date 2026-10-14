"""`llmbytes exercise` and the layout of `exercises/`.

The exercise tests themselves run in CI against `exercises/solutions/` (LLMBYTES_SOLUTIONS=1).
These check that every exercise has the three files it needs, and that a stub and its solution
define the same functions, so a learner who fills in the stub can pass.
"""

from __future__ import annotations

import ast

from llmbytes import exercise


def _functions(path):
    return {n.name for n in ast.parse(path.read_text(encoding="utf-8")).body if isinstance(n, ast.FunctionDef)}


def test_the_seven_exercises():
    assert list(exercise.available()) == ["1b", "2b", "3b", "4b", "5b", "6b", "8b"]


def test_each_has_a_stub_and_a_solution_with_the_same_functions():
    for ident, (name, _) in exercise.available().items():
        stub = exercise.EXERCISES / "stubs" / f"{name}.py"
        solution = exercise.EXERCISES / "solutions" / f"{name}.py"
        assert stub.exists() and solution.exists(), ident
        assert _functions(stub) == _functions(solution), ident


def test_unknown_exercise_is_refused(capsys):
    assert exercise.run("7b") == 2
    assert "1b, 2b, 3b, 4b, 5b, 6b, 8b" in capsys.readouterr().err
