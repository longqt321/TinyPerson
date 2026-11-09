"""Stable-append CSV logging for per-epoch metrics.

Every training callback that records per-epoch numbers appends to a CSV. The
header must stay stable across epochs: callbacks can be re-created on resume or
when a two-stage run chains into stage 2, and the set of logged keys can change
between epochs. `append_epoch_csv_row` freezes the fieldnames on the first
write (or adopts the existing file header), drops unexpected keys with
`extrasaction="ignore"`, and fills missing keys with empty cells so the columns
never misalign.
"""

import csv
from pathlib import Path


def append_epoch_csv_row(
    path: Path, epoch: int, rows: dict[str, float], state: dict
) -> None:
    """Append one epoch row to `path`, keeping the header stable across epochs.

    `state` is a per-callback dict holding the frozen fieldnames.
    """
    fieldnames = state.get("fieldnames")
    if fieldnames is None:
        if path.is_file():
            with path.open(newline="") as file:
                first_line = file.readline().strip()
            if first_line:
                fieldnames = first_line.split(",")
        if not fieldnames:
            fieldnames = ["epoch", *sorted(rows)]
        state["fieldnames"] = fieldnames
    with path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        if file.tell() == 0:
            writer.writeheader()
        writer.writerow({"epoch": epoch, **rows})
