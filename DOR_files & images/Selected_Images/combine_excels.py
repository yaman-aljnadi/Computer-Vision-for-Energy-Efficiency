#!/usr/bin/env python3
"""
Combine many CSV files into one dataset, then export to Excel or CSV.

Usage examples:
  # Basic: combine all CSVs in a folder -> Excel
  python combine_csvs.py --input-dir "path/to/folder" --output "combined.xlsx"

  # Write CSV instead of Excel
  python combine_csvs.py --input-dir "path/to/folder" --output "combined.csv"
  python combine_excels.py --input-dir "Chippewa_Selected_Images_MMdetection_Original" --output "Combined_Excel_Results/combined_MMDetection_0.4.csv"

  # Recurse subfolders and limit to pattern
  python combine_csvs.py --input-dir "/data" --pattern "*.csv" --recursive --output "combined.xlsx"

  # If files are large, stream in chunks and de-duplicate rows
  python combine_csvs.py --input-dir "/data" --chunksize 100000 --dedupe --output "combined.csv"
"""

import argparse
from pathlib import Path
import pandas as pd
import csv

def sniff_delimiter(sample_bytes: bytes, default=","):
    """Best-effort delimiter detection from a bytes sample."""
    try:
        sample = sample_bytes.decode("utf-8", errors="ignore")
        dialect = csv.Sniffer().sniff(sample, delimiters=[",",";","|","\t"])
        return dialect.delimiter
    except Exception:
        return default

def read_csv_safely(path: Path, chunksize=None) -> pd.DataFrame | list[pd.DataFrame]:
    """
    Read a CSV with best-effort delimiter + encoding handling.
    Returns a DataFrame (no chunksize) or an iterable/list of DataFrames (with chunksize).
    """
    # Peek first ~64 KB to guess delimiter
    with open(path, "rb") as f:
        head = f.read(65536)
    delimiter = sniff_delimiter(head, default=",")

    read_kwargs = dict(
        sep=delimiter,
        dtype=object,          # keep types flexible across files
        engine="python",       # robust with weird delimiters/quotes
        on_bad_lines="skip",   # skip malformed rows
    )

    # Try a couple of common encodings
    for enc in ("utf-8-sig", "utf-8", "latin1"):
        try:
            if chunksize:
                return pd.read_csv(path, encoding=enc, chunksize=chunksize, **read_kwargs)
            else:
                return pd.read_csv(path, encoding=enc, **read_kwargs)
        except Exception:
            continue

    # Last-resort: return empty if unreadable
    print(f"[WARN] Skipping '{path}': could not decode with common encodings.")
    return pd.DataFrame()

def main():
    ap = argparse.ArgumentParser(description="Combine CSV files into a single dataset.")
    ap.add_argument("--input-dir", required=True, help="Folder containing CSV files")
    ap.add_argument("--pattern", default="*.csv", help="Glob pattern, e.g. '*.csv'")
    ap.add_argument("--output", default="combined.xlsx", help="Output path (.xlsx or .csv)")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders")
    ap.add_argument("--dedupe", action="store_true", help="Drop exact duplicate rows after combining")
    ap.add_argument("--chunksize", type=int, default=None, help="Read CSVs in chunks (e.g., 100000) for large files")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    files = sorted(input_dir.rglob(args.pattern) if args.recursive else input_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files matched pattern '{args.pattern}' in {input_dir}")

    print(f"Found {len(files)} file(s). Reading…")

    frames = []
    total_rows = 0

    for f in files:
        chunks = read_csv_safely(f, chunksize=args.chunksize)

        if isinstance(chunks, pd.DataFrame):
            df = chunks
            if df.empty:
                continue
            df["source_file"] = f.name
            frames.append(df)
            total_rows += len(df)
        elif hasattr(chunks, "__iter__"):  # chunked iterator
            for chunk in chunks:
                if chunk.empty:
                    continue
                chunk["source_file"] = f.name
                frames.append(chunk)
                total_rows += len(chunk)
        else:
            # empty or unreadable
            continue

    if not frames:
        raise SystemExit("No readable CSV content found.")

    combined = pd.concat(frames, ignore_index=True, sort=False)

    if args.dedupe:
        before = len(combined)
        combined = combined.drop_duplicates()
        print(f"De-duplicated rows: {before - len(combined)} removed")

    out = Path(args.output)
    if out.suffix.lower() == ".csv":
        combined.to_csv(out, index=False)
        print(f"Done! Wrote {len(combined):,} rows and {combined.shape[1]} columns to '{out}'.")
    else:
        # Default to Excel (xlsx)
        with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
            combined.to_excel(writer, index=False, sheet_name="Combined")
            ws = writer.sheets["Combined"]
            ws.freeze_panes(1, 0)
            for i, col in enumerate(combined.columns):
                width = max(12, min(50, len(str(col)) + 4))
                ws.set_column(i, i, width)
        print(f"Done! Wrote {len(combined):,} rows and {combined.shape[1]} columns to '{out}'.")

if __name__ == "__main__":
    main()
