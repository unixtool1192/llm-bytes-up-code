"""Helpers shared by the lesson scripts: open a GGUF model and read what is inside it."""

import sys

# The lessons print tokens such as 'Ġcat' and '猫'. On Windows, Python writes a redirected
# stdout (`| tee`, `> file`) in the system code page, usually cp1252, which cannot encode them
# and stops with UnicodeEncodeError; `llmbytes check -` would likewise misread UTF-8 from a
# pipe. Every lesson imports this package, so switching the standard streams to UTF-8 here
# makes them behave as they do on Linux and macOS. It is the same as setting PYTHONUTF8=1.
if sys.platform == "win32":
    for _stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8")

from .model import GGUFModel, rms_norm, silu, softmax, cosine  # noqa: E402

__all__ = ["GGUFModel", "rms_norm", "silu", "softmax", "cosine"]
