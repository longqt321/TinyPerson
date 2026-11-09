"""Baseline JSONL summarisation for error-analysis comparison.

``samples.jsonl`` stores one full record per image (ground truth, predictions,
summary). ``_compare_samples`` only needs ``image.basename``,
``summary.misses`` and ``summary.false_positives`` from the baseline file, so
parsing the whole file into dicts is wasted work: on a 46.5 MB file the full
parse takes ~1.4 s and materialises ~140k dicts.

This module parses the file in parallel chunks and keeps only the two summary
fields per image. Measured on the real 56.5 MB ``errors.jsonl`` shape:
sequential full parse 844 ms vs 303 ms with a 4-worker interpreter pool
(2.79x), provided the worker aggregates and returns small results.

InterpreterPoolExecutor worker contract (Python 3.14, verified empirically):

- Worker functions are shared *by value* (code object). Any reference to a
  module-level global -- including constants -- raises ``NotShareableError``.
  The worker below therefore uses only locals and imports ``json`` inside its
  body.
- The subinterpreter cannot load numpy/torch/PIL C extensions, so this module
  must stay stdlib-only and must not be imported by modules that need those at
  call time inside workers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Payloads below this size parse faster sequentially (measured: 0.83 MB is
# 2.2x slower with a pool, 5 MB is 1.4x faster, 25 MB is 2.4x faster).
PARALLEL_MIN_BYTES = 5_000_000
# samples.jsonl lines are large (~59 KB each, 786 lines for 46.5 MB), so
# chunking must be byte-based, not line-count-based.
CHUNK_BYTES = 1_000_000
MAX_WORKERS = 4


def _split_at_line_boundaries(text: str, target_bytes: int) -> list[str]:
    """Split ``text`` into chunks of ~``target_bytes``, never mid-line."""
    chunks: list[str] = []
    start = 0
    total = len(text)
    while start < total:
        end = min(start + target_bytes, total)
        if end < total:
            newline = text.find("\n", end)
            end = total if newline == -1 else newline + 1
        chunks.append(text[start:end])
        start = end
    return chunks


def summarize_chunk(chunk: str):
    """Parse one JSONL chunk of sample records, keeping only comparison fields.

    Returns ``(keys_uniform, first_dataset_key, {basename: (misses, fps)})``.
    ``keys_uniform`` is False when records disagree on ``dataset_key``.
    """
    import json

    by_image = {}
    keys_uniform = True
    first_key = None
    seen_key = False
    for line in chunk.splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        key = record.get("dataset_key")
        if not seen_key:
            first_key = key
            seen_key = True
        elif key != first_key:
            keys_uniform = False
        summary = record["summary"]
        by_image[record["image"]["basename"]] = (
            summary["misses"],
            summary["false_positives"],
        )
    return keys_uniform, first_key, by_image


def load_baseline_summaries(
    baseline_path: str | Path,
) -> tuple[bool, Any, dict[str, tuple[int, int]]]:
    """Extract per-image ``(misses, false_positives)`` from a samples JSONL.

    Returns ``(keys_uniform, first_dataset_key, summaries)`` where
    ``summaries`` maps basename to ``(misses, false_positives)``. Duplicate
    basenames keep the last record, matching the dict-comprehension behaviour
    this replaces.
    """
    path = Path(baseline_path)
    text = path.read_text(encoding="utf-8")
    if path.stat().st_size < PARALLEL_MIN_BYTES:
        return summarize_chunk(text)

    chunks = _split_at_line_boundaries(text, CHUNK_BYTES)
    from concurrent.futures import InterpreterPoolExecutor

    with InterpreterPoolExecutor(max_workers=MAX_WORKERS) as pool:
        results = list(pool.map(summarize_chunk, chunks))

    merged: dict[str, tuple[int, int]] = {}
    keys_uniform = True
    first_key: Any = None
    seen_key = False
    for uniform, key, part in results:
        if not uniform:
            keys_uniform = False
        if not seen_key:
            first_key, seen_key = key, True
        elif key != first_key:
            keys_uniform = False
        merged.update(part)
    return keys_uniform, first_key, merged
