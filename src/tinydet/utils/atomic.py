"""Atomic text writes for artifacts produced while training is running.

Uses Python's standard ``tempfile`` + ``os.fsync`` + ``os.replace`` idiom
(``atomicwrites`` is deprecated in favor of ``os.replace`` on modern Python).
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any


@contextmanager
def _atomic_open(path: str | Path, *, fsync: bool = True) -> Iterator[IO[str]]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            yield stream
            stream.flush()
            if fsync:
                os.fsync(stream.fileno())
        os.replace(tmp_path, destination)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def atomic_write_text(path: str | Path, text: str, *, fsync: bool = True) -> None:
    """Write ``text`` to ``path`` atomically, creating parent directories."""
    with _atomic_open(path, fsync=fsync) as stream:
        stream.write(text)


def atomic_write_jsonl(
    path: str | Path, records: Iterable[dict[str, Any]], *, fsync: bool = True
) -> None:
    """Stream records as JSON Lines into ``path`` atomically in O(1) memory."""
    with _atomic_open(path, fsync=fsync) as stream:
        for record in records:
            stream.write(json.dumps(record))
            stream.write("\n")
