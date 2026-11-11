#!/usr/bin/env python3
"""Run the protocol checker that the task package ships.

``protocol_check/run.py`` implements the wire format of the judge: the JSON
Lines protocol, the ``{"type":"ready"}`` handshake, one response per request,
the ``--restart`` and ``--mode ann`` directives and the assertion semantics of
the ``.result`` files.  This wrapper only builds its command line, so that the
local self test and the judge cannot drift apart.

The checker uses ``os.read`` on a pipe and ``os.killpg``, so it needs a POSIX
system.  On Linux it can be run directly, on Windows it has to run inside a
container::

    docker build -f Dockerfile.check -t mvdb-checker:local .
    docker run --rm -v mvdb-check-out:/out mvdb-checker:local \\
        python tools/run_checker.py --output /out/check1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
sys.path.insert(0, str(_REPO_ROOT / "protocol_check"))

import run as checker  # noqa: E402  (the task package module)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_checker.py",
        description="Run the public cases with the protocol checker of the package.",
    )
    parser.add_argument(
        "--case", action="append", help="public case id; repeat to select several"
    )
    parser.add_argument("--image", help="run the cases against a built Docker image")
    parser.add_argument(
        "--program",
        default="{} -m minivectordb --data-dir {{data_dir}}".format(sys.executable),
        help="the local program, {data_dir} is replaced by the data directory",
    )
    parser.add_argument("--output", required=True, help="directory for the report")
    arguments = parser.parse_args(argv)

    checker_argv = ["--output", arguments.output]
    if arguments.image:
        checker_argv += ["--image", arguments.image]
    else:
        program = arguments.program.replace("{data_dir}", "{data_dir}")
        checker_argv += ["--command", json.dumps(program.split())]
    for case in arguments.case or []:
        checker_argv += ["--case", case]
    return checker.main(checker_argv)


if __name__ == "__main__":
    sys.exit(main())
