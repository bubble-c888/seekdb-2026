"""Entry point: ``python -m minivectordb --data-dir /data``.

The program stays in the foreground, reads SQL statements from stdin and
writes one JSON result object per statement to stdout.  No network socket is
opened, exactly as the task statement requires.

The retrieval mode (``exact`` or ``ann``) can be selected in several ways so
that the test driver finds a channel that works for it:

* command line: ``--mode ann`` / ``--search-mode ann`` / ``--ann``
* environment: ``MINIVECTORDB_MODE``, ``MINIVECTORDB_SEARCH_MODE``,
  ``SEARCH_MODE``, ``MODE``, ``VECTOR_SEARCH_MODE``, ``RETRIEVAL_MODE``
* inside the session: ``SET search_mode = 'ann';``
* a driver directive on stdin: ``--mode ann``
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .engine.database import Database
from .server import Server

#: Environment variables that may carry the initial retrieval mode.
MODE_ENV_VARS = (
    "MINIVECTORDB_MODE",
    "MINIVECTORDB_SEARCH_MODE",
    "SEARCH_MODE",
    "MODE",
    "VECTOR_SEARCH_MODE",
    "RETRIEVAL_MODE",
)


def initial_mode(default: str = "exact") -> str:
    """Read the retrieval mode from the environment."""
    for name in MODE_ENV_VARS:
        value = os.environ.get(name)
        if value:
            value = value.strip().lower()
            if value in ("exact", "ann"):
                return value
    return default


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
        "--search-mode",
        "--retrieval-mode",
        dest="mode",
        default=initial_mode(),
        choices=["exact", "ann"],
        help="retrieval mode used for vector TopK queries (default: exact)",
    )
    parser.add_argument(
        "--ann",
        dest="ann",
        action="store_true",
        help="shortcut for --mode ann",
    )
    parser.add_argument(
        "--exact",
        dest="exact",
        action="store_true",
        help="shortcut for --mode exact (the default)",
    )
    parser.add_argument(
        "--ready-marker",
        default=os.environ.get("MINIVECTORDB_READY_MARKER", "stdout"),
        choices=["stdout", "stderr", "both", "none", "off"],
        help=(
            "where the readiness marker is written; it is only sent when the "
            "driver did not send a statement yet (default: stdout)"
        ),
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    mode = "ann" if args.ann else "exact" if args.exact else args.mode.lower()
    data_dir = Path(args.data_dir)
    database = Database(data_dir)
    database.mode = mode
    database.open()
    server = Server(database, ready_target=args.ready_marker)
    return server.run()


if __name__ == "__main__":
    sys.exit(main())
