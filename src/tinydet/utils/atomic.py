"""Atomic text writes for artifacts produced while training is running.

A container can die at any moment (spot preemption, network loss, OOM kill). A
plain ``Path.write_text`` truncates the destination first, so a crash mid-write
leaves an unreadable JSON artifact. Writing to a sibling temporary file and
``os.replace``-ing it into place is atomic on POSIX: readers see either the old
file or the complete new one, never a half-written file.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def _temporary_sibling(destination: Path) -> tuple[int, str]:
    return tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )


def atomic_write_text(path: str | Path, text: str, *, fsync: bool = True) -> None:
    """Write ``text`` to ``path`` atomically, creating parent directories."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = _temporary_sibling(destination)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            file.write(text)
            file.flush()
            if fsync:
                os.fsync(file.fileno())
        os.replace(temporary, destination)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def atomic_write_jsonl(
    path: str | Path, records: Iterable[dict[str, Any]], *, fsync: bool = True
) -> None:
    """Stream records as JSON Lines into ``path`` atomically.

    Records are serialised one at a time, so memory stays O(1) regardless of how
    many samples the analysis produced.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = _temporary_sibling(destination)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record))
                file.write("\n")
            file.flush()
            if fsync:
                os.fsync(file.fileno())
        os.replace(temporary, destination)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
