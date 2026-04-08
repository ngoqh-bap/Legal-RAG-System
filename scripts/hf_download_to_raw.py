"""
Download Hugging Face datasets into data/raw/ for ingestion.

By default this exports each split to JSONL under:
  data/raw/hf/<dataset_id>/<config>/<split>.jsonl

If you pass --text-column, it also exports a concatenated .txt file for that split.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def _sanitize_path_part(s: str) -> str:
    return s.replace("\\", "_").replace("/", "__").replace(":", "_").strip()


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_text(path: Path, rows: list[dict[str, Any]], text_column: str) -> None:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            v = row.get(text_column)
            if v is None:
                continue
            if isinstance(v, list):
                for item in v:
                    if item is not None:
                        f.write(str(item).strip() + "\n")
                continue
            f.write(str(v).strip() + "\n\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download HF datasets into data/raw/")
    parser.add_argument("--dataset", required=True, help='Dataset id, e.g. "th1nhng0/vietnamese-legal-documents"')
    parser.add_argument("--config", default=None, help="Optional dataset config name")
    parser.add_argument("--split", default=None, help='Optional split name (default: all splits). Use "*" for all.')
    parser.add_argument(
        "--out-dir",
        default=os.path.join("data", "raw", "hf"),
        help='Output base directory (default: "data/raw/hf")',
    )
    parser.add_argument(
        "--text-column",
        default=None,
        help="If set, also export concatenated .txt using this column (e.g. 'text', 'content').",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap per split (useful for quick demos).",
    )

    args = parser.parse_args()

    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Missing dependency. Install with:\n"
            '  pip install -e ".[hf]"\n'
            "Then re-run this script."
        ) from e

    ds = load_dataset(args.dataset, args.config) if args.config else load_dataset(args.dataset)

    out_base = Path(args.out_dir) / _sanitize_path_part(args.dataset) / _sanitize_path_part(args.config or "default")

    if isinstance(ds, dict):
        split_names = list(ds.keys())
        if args.split and args.split != "*":
            split_names = [args.split]
        for split in split_names:
            table = ds[split]
            rows = table.to_list()
            if args.max_rows is not None:
                rows = rows[: args.max_rows]

            jsonl_path = out_base / f"{_sanitize_path_part(split)}.jsonl"
            _write_jsonl(jsonl_path, rows)

            if args.text_column:
                txt_path = out_base / f"{_sanitize_path_part(split)}.txt"
                _write_text(txt_path, rows, args.text_column)
            print(f"Wrote {len(rows)} rows: {jsonl_path.as_posix()}")
        return 0

    # Single split dataset object
    rows = ds.to_list()
    if args.max_rows is not None:
        rows = rows[: args.max_rows]
    jsonl_path = out_base / "data.jsonl"
    _write_jsonl(jsonl_path, rows)
    if args.text_column:
        txt_path = out_base / "data.txt"
        _write_text(txt_path, rows, args.text_column)
    print(f"Wrote {len(rows)} rows: {jsonl_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

