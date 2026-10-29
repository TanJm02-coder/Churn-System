"""Reproducibly download the IBM Telco Customer Churn dataset from OpenML.

Writes the raw CSV to ``data/raw/telco_churn.csv`` and records the OpenML
version plus a SHA-256 checksum in ``data/raw/DATA_MANIFEST.json`` so the exact
data version is reproducible. Idempotent: skips the download if the CSV already
exists, unless ``--force`` is passed.

Usage:
    python scripts/download_data.py [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime

from sklearn.datasets import fetch_openml

from churn.config import DATA_RAW, MANIFEST, OPENML_DATA_ID, RAW_CSV


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download(force: bool = False) -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    if RAW_CSV.exists() and not force:
        print(f"[skip] {RAW_CSV} already exists (use --force to re-download).")
        return

    print(f"[fetch] OpenML dataset id={OPENML_DATA_ID} ...")
    cache_dir = DATA_RAW / ".openml_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    bunch = fetch_openml(
        data_id=OPENML_DATA_ID, as_frame=True, parser="auto", data_home=str(cache_dir)
    )
    df = bunch.frame

    df.to_csv(RAW_CSV, index=False)
    print(f"[write] {RAW_CSV}  ({len(df)} rows, {df.shape[1]} cols)")

    manifest = {
        "source": "OpenML",
        "data_id": OPENML_DATA_ID,
        "openml_version": 1,
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "columns": list(df.columns),
        "sha256": _sha256(RAW_CSV),
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"[write] {MANIFEST}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-download even if present.")
    args = parser.parse_args()
    download(force=args.force)


if __name__ == "__main__":
    main()
