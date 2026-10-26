"""Entry point for the local nam-redactor web app."""
from __future__ import annotations

import argparse
import sys
import threading
import webbrowser
from pathlib import Path

import uvicorn

from .app import create_app
from .storage import DATA_DIR_ENV, default_data_dir

#: Loopback only. There is deliberately no --host flag: this service holds
#: health documents, so binding it to a routable interface must not be one
#: typo away.
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nam-redact-web",
        description="Local web app to redact NAM (RAMQ health numbers) from PDF files.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port to listen on (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not open a browser window on startup",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=(
            "Where documents are saved (default: per-user app data, "
            f"or ${DATA_DIR_ENV})"
        ),
    )
    args = parser.parse_args(argv)

    url = f"http://{DEFAULT_HOST}:{args.port}"
    data_dir = args.data_dir or default_data_dir()
    app = create_app(dir_root=data_dir)

    print(f"nam-redactor is running at {url}")
    print(f"Documents are saved in {data_dir}")
    print("Press Ctrl+C to stop.")
    if not args.no_open:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()

    try:
        # workers=1 is required: job state and the SSE fan-out live in memory,
        # and a second worker would see an empty job registry. Paddle's thread
        # env vars are set by the nam_redactor package import above.
        uvicorn.run(
            app,
            host=DEFAULT_HOST,
            port=args.port,
            workers=1,
            log_level="warning",
        )
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
