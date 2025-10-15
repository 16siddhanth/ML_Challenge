"""Utility script to download product images locally so they can be reused offline."""
from __future__ import annotations

import argparse
import logging
from collections import OrderedDict
from pathlib import Path

import pandas as pd

from .utils import download_image_with_retry, ensure_directory, setup_logging, ensure_project_structure


def download_images(csv_paths: list[Path], output_dir: Path, limit: int | None = None) -> None:
    """Download images listed across one or more CSVs to a local directory."""

    ensure_project_structure()
    ensure_directory(output_dir)

    records: OrderedDict[str, str] = OrderedDict()
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        for row in df.itertuples():
            url = getattr(row, "image_link", "")
            sample_id = str(getattr(row, "sample_id"))
            if sample_id and url and sample_id not in records:
                records[sample_id] = url

    total = len(records) if limit is None else min(limit, len(records))
    logging.info("Downloading images for %s unique sample_ids", total)

    for idx, (sample_id, url) in enumerate(records.items(), start=1):
        if limit is not None and idx > limit:
            break
        destination = output_dir / f"{sample_id}.jpg"
        if destination.exists():
            continue
        download_image_with_retry(url, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download product images referenced in CSV files")
    parser.add_argument("--csv", type=Path, nargs="+", required=True, help="One or more CSV files with image_link column")
    parser.add_argument("--output", type=Path, default=Path("data/raw/images"), help="Destination folder for images")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of rows to process")
    args = parser.parse_args()

    setup_logging()
    download_images(args.csv, args.output, args.limit)


if __name__ == "__main__":
    main()
