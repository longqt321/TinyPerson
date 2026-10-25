"""Append-only JSONL diagnostics sink."""

import json
import math
from pathlib import Path


class JsonlSink:
    def __init__(self, path: Path, flush_records: int = 10) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("a", encoding="utf-8")
        self.flush_records = flush_records
        self.pending = 0

    def write(self, record: dict) -> None:
        values = dict(record)
        values["metrics"] = {
            key: _json_number(value) for key, value in values["metrics"].items()
        }
        self.file.write(json.dumps(values, allow_nan=False, sort_keys=True) + "\n")
        self.pending += 1
        if self.pending >= self.flush_records:
            self.file.flush()
            self.pending = 0

    def close(self) -> None:
        if not self.file.closed:
            self.file.flush()
            self.file.close()


def _json_number(value):
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None
