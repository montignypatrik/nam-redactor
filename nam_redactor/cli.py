"""Command-line interface for nam-redactor."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Threading env vars
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("FLAGS_use_mkldnn", "0")

from .redactor import redact_pdf


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Redact NAM (RAMQ health numbers) from PDF files."
    )
    parser.add_argument("input_pdf", type=Path, help="Path to input PDF file")
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Rasterization DPI (default: 200, use 150 for low-memory)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Path to output redacted PDF (default: <input>.redacted.pdf)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Write detected boxes to JSON (boxes + confidence only) and preview PNG per page",
    )

    args = parser.parse_args(argv)

    input_path = args.input_pdf.resolve()
    if not input_path.exists():
        print(f"Error: Input file does not exist: {input_path}", file=sys.stderr)
        return 1

    if args.out:
        out_path = args.out.resolve()
    else:
        out_path = input_path.parent / f"{input_path.stem}.redacted.pdf"

    print(f"Processing '{input_path.name}' at {args.dpi} DPI...")
    try:
        total, per_page, _ = redact_pdf(
            input_path=input_path,
            out_path=out_path,
            dpi=args.dpi,
            debug=args.debug,
        )
    except Exception as exc:
        print(f"Error processing PDF: {exc}", file=sys.stderr)
        return 1

    for idx, count in enumerate(per_page, start=1):
        print(f"Page {idx}: {count} NAM(s) redacted")
    print(f"Total: {total} NAM(s) redacted")
    print(f"Output saved to: {out_path}")
    if args.debug:
        debug_dir = out_path.parent / f"{out_path.stem}_debug"
        print(f"Debug artifacts saved to: {debug_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
