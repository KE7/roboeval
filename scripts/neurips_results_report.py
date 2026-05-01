#!/usr/bin/env python3
"""Generate interim NeurIPS LIBERO-Infinity result tables.

The script reads completed roboeval benchmark JSON files and writes task-level
and suite-level rows with Wilson 95% confidence intervals. Suite rows are
computed as task means and do not pool episodes across tasks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from roboeval.results.stats import (
    find_result_files,
    format_markdown_table,
    latest_summaries_per_cell,
    rows_for_summaries,
    summarize_result_file,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        default="results/neurips2026",
        help="Result directory or single result JSON to scan.",
    )
    parser.add_argument(
        "--out",
        default="results/neurips2026/interim_tables.md",
        help="Output path. Suffix controls format: .md, .csv, or .json.",
    )
    parser.add_argument(
        "--include-all-runs",
        action="store_true",
        help="Include every completed result JSON instead of only the latest per cell.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = find_result_files(args.results_root)
    summaries = [summarize_result_file(path) for path in paths]
    if not args.include_all_runs:
        summaries = latest_summaries_per_cell(summaries)
    rows = rows_for_summaries(summaries)

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".csv":
        write_csv(rows, output)
    elif output.suffix == ".json":
        output.write_text(json.dumps(rows, indent=2))
    else:
        if rows:
            output.write_text(format_markdown_table(rows))
        else:
            output.write_text(
                "# Interim NeurIPS LIBERO-Infinity Results\n\n"
                "No completed result JSON files were found. No runs are fabricated.\n\n"
                "No videos were verified because no episodes ran.\n"
            )

    print(f"Found {len(paths)} completed result file(s). Wrote {len(rows)} row(s) to {output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
