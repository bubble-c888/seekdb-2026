"""Entry point: ``python -m minivectordb --data-dir /data``.

The program stays in the foreground, reads SQL statements from stdin and
writes one JSON result object per statement to stdout.  No network socket is
opened, exactly as the task statement requires.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .engine.database import Database
from .server import Server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="minivectordb",
        description="MiniVectorDB - a small vector database with SQL access",
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("MINIVECTORDB_DATA_DIR", "/data"),
        help="directory that holds the database files (default: /data)",
    )
    parser.add_argument(
        "--mode",
        default=os.environ.get("MINIVECTORDB_MODE", "exact"),
        choices=["exact", "ann"],
        help="retrieval mode used for vector TopK queries (default: exact)",
    )
    parser.add_argument(
        "--ready-marker",
        default=os.environ.get("MINIVECTORDB_READY_MARKER", "stdout"),
        choices=["stdout", "stderr", "both", "none", "off"],
        help="where the readiness marker is written (default: stdout)",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    data_dir = Path(args.data_dir)
    database = Database(data_dir)
    database.mode = args.mode.lower()
    database.open()
    server = Server(database)
    return server.run()


if __name__ == "__main__":
    sys.exit(main())
