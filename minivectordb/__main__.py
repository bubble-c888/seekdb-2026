"""Entry point: ``python -m minivectordb --data-dir /data``.

The program stays in the foreground, speaks the JSON Lines protocol of the
task package on stdin/stdout and opens no network socket.

The retrieval mode (``exact`` or ``ann``) is chosen per request by the
judge (``search_mode``).  The options below only set the mode that is used
before the first request arrives, so that a driver that starts the program
with a mode also works.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# The linear algebra of numpy is single threaded on purpose.  The judge gives
# the container two cores, and the matrix product of the library spawns a
# thread pool for a matrix of this size: every call then pays for waking the
# threads up again, which measured 1.1 ms per call in the evaluation
# container against 0.03 ms with one thread.  The setting has to be in place
# before numpy is imported, because the library reads it when it is loaded.
for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

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

#: Environment variables that may carry the data directory.
DATA_DIR_ENV_VARS = ("MINIVECTORDB_DATA_DIR", "VECTORDB_DATA_DIR")


def initial_mode(default: str = "exact") -> str:
    """Read the retrieval mode from the environment."""
    for name in MODE_ENV_VARS:
        value = os.environ.get(name)
        if value:
            value = value.strip().lower()
            if value in ("exact", "ann"):
                return value
    return default


def initial_data_dir() -> str:
    """Read the data directory from the environment."""
    for name in DATA_DIR_ENV_VARS:
        value = os.environ.get(name)
        if value:
            return value
    return "/data"


def force_utf8_streams() -> None:
    """Make the protocol streams UTF-8, whatever the console code page is.

    The protocol is UTF-8 by definition.  On a Windows host the standard
    streams would otherwise use the code page of the console, and a Chinese
    text would reach the judge as mojibake.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except (AttributeError, ValueError):  # pragma: no cover - defensive
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="minivectordb",
        description="MiniVectorDB - a small vector database with SQL access",
    )
    parser.add_argument(
        "--data-dir",
        default=initial_data_dir(),
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
            "where the startup handshake is written; the judge expects it on "
            "stdout, 'none' only exists for local debugging (default: stdout)"
        ),
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    force_utf8_streams()
    mode = "ann" if args.ann else "exact" if args.exact else args.mode.lower()
    data_dir = Path(args.data_dir)
    database = Database(data_dir)
    database.mode = mode
    database.open()
    server = Server(database, ready_target=args.ready_marker)
    return server.run()


if __name__ == "__main__":
    sys.exit(main())
