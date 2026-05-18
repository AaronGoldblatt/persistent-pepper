"""Small logging helpers for long-running Google Colab training jobs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable


def append_training_metrics_to_csv(
    csv_file_path: Path,
    fieldnames: Iterable[str],
    metric_row: Dict[str, object],
) -> None:
    """Append one metrics row, creating the CSV header on first write."""
    csv_file_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames_as_list = list(fieldnames)
    should_write_header = (
        not csv_file_path.exists()
        or csv_file_path.stat().st_size == 0
    )
    with csv_file_path.open("a", newline="") as csv_file_handle:
        csv_writer = csv.DictWriter(
            csv_file_handle,
            fieldnames=fieldnames_as_list,
        )
        if should_write_header:
            csv_writer.writeheader()
        csv_writer.writerow(metric_row)
