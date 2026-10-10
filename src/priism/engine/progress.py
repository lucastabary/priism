"""Progress reporting: JSON lines on stdout for MyMixxx, short text on stderr for a person.

With ``--progress jsonl`` every event is one JSON object per line on stdout, flushed at once
so a ``QProcess`` reading the pipe sees it immediately:

    {"event": "progress", "stage": "model:htdemucs_ft", "fraction": 0.42}
    {"event": "warning", "message": "..."}
    {"event": "done", ...}
    {"event": "error", "message": "..."}

Without it, stdout stays quiet and a person gets readable lines on stderr.
"""

from __future__ import annotations

import json
import sys
from typing import Any, TextIO


class Reporter:
    def __init__(self, mode: str | None = None, out: TextIO | None = None, err: TextIO | None = None):
        if mode not in (None, "jsonl"):
            raise ValueError(f"unknown progress mode {mode!r}")
        self.mode = mode
        self.out = out or sys.stdout
        self.err = err or sys.stderr
        self._last: tuple[str, int] | None = None

    def _emit(self, event: dict[str, Any]) -> None:
        self.out.write(json.dumps(event, ensure_ascii=False) + "\n")
        self.out.flush()

    def progress(self, stage: str, fraction: float) -> None:
        fraction = min(max(float(fraction), 0.0), 1.0)
        if self.mode == "jsonl":
            self._emit({"event": "progress", "stage": stage, "fraction": round(fraction, 4)})
            return
        # Text mode: one line per stage and per 10 %, not one per model chunk.
        key = (stage, int(fraction * 10))
        if key != self._last:
            self._last = key
            self.err.write(f"{stage}: {fraction:.0%}\n")
            self.err.flush()

    def stage(self, stage: str):
        """A callback ``fraction -> None`` bound to one stage, for code that knows nothing of stages."""
        return lambda fraction: self.progress(stage, fraction)

    def warning(self, message: str) -> None:
        if self.mode == "jsonl":
            self._emit({"event": "warning", "message": message})
        else:
            self.err.write(f"warning: {message}\n")

    def done(self, **fields: Any) -> None:
        if self.mode == "jsonl":
            self._emit({"event": "done", **fields})
        else:
            self.err.write("done: " + ", ".join(f"{k}={v}" for k, v in fields.items()) + "\n")

    def error(self, message: str) -> None:
        """Errors always reach stderr (the contract); in jsonl mode they are also an event."""
        if self.mode == "jsonl":
            self._emit({"event": "error", "message": message})
        self.err.write(f"error: {message}\n")
        self.err.flush()
