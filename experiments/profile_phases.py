#!/usr/bin/env python3
"""Where does the time of the benchmark go, in the evaluation container?

The program is driven the way the judge drives it: the requests arrive as
JSON Lines on stdin and the answers are written to stdout.  The script times
the write phase and the query phase separately and then profiles them.

Run it in the container::

    docker run --rm --entrypoint python --network=none --cpus=2 --memory=512m \
        --read-only --cap-drop=ALL --security-opt=no-new-privileges \
        -e TMPDIR=/data -v /path/to/repo:/repo:ro -v mvdb-prof:/data \
        mvdb-change:local /repo/experiments/profile_phases.py
"""
from __future__ import annotations

import cProfile
import io
import json
import pstats
import random
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, "/app")

import numpy as np  # noqa: E402

from minivectordb.engine.database import Database  # noqa: E402
from minivectordb.engine.executor import Executor  # noqa: E402
from minivectordb.sqlparse import parse  # noqa: E402

ROWS = 10000
DIM = 128
BATCH = 100
QUERIES = 1000


def main() -> int:
    random.seed(20261111)
    vectors = [[random.uniform(-1.0, 1.0) for _ in range(DIM)] for _ in range(ROWS)]
    groups = [random.randrange(3) for _ in range(ROWS)]

    directory = Path("/data/phases")
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True, exist_ok=True)

    database = Database(directory)
    database.open()
    executor = Executor(database)

    # ---------------------------------------------------------------- write
    statements = []
    for start in range(0, ROWS, BATCH):
        values = ",".join(
            "({},{},'[{}]')".format(
                index,
                groups[index],
                ",".join("%.6f" % value for value in vectors[index]),
            )
            for index in range(start, min(start + BATCH, ROWS))
        )
        statements.append("INSERT INTO docs VALUES " + values)

    profiler = cProfile.Profile()
    profiler.enable()
    began = time.perf_counter()
    executor.execute(parse("CREATE TABLE docs (id BIGINT PRIMARY KEY, g BIGINT, v VECTOR(128));"))
    for statement in statements:
        executor.execute(parse(statement))
    write_seconds = time.perf_counter() - began
    profiler.disable()
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats("tottime").print_stats(12)
    print("write {:.3f} s".format(write_seconds))
    for line in stream.getvalue().splitlines():
        if "{" in line and "function calls" in line:
            continue
        print(line)

    # ---------------------------------------------------------------- query
    queries = []
    for number in range(QUERIES):
        vector = [random.uniform(-1.0, 1.0) for _ in range(DIM)]
        literal = "[" + ",".join("%.6f" % value for value in vector) + "]"
        if number % 2 == 0:
            sql = "SELECT id FROM docs WHERE g = {} ORDER BY L2_DISTANCE(v,'{}') LIMIT 10".format(
                random.randrange(3), literal
            )
        else:
            sql = "SELECT id FROM docs ORDER BY L2_DISTANCE(v,'{}') LIMIT 10".format(literal)
        queries.append(sql)

    for sql in queries[:5]:
        executor.execute(parse(sql))
    profiler = cProfile.Profile()
    profiler.enable()
    began = time.perf_counter()
    for sql in queries:
        executor.execute(parse(sql))
    query_seconds = time.perf_counter() - began
    profiler.disable()
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats("tottime").print_stats(12)
    print("\nquery {:.3f} ms".format(query_seconds / len(queries) * 1000))
    for line in stream.getvalue().splitlines():
        if "{" in line and "function calls" in line:
            continue
        print(line)

    # ------------------------------------------------- the protocol overhead
    payload = json.dumps({"id": 1, "op": "execute", "sql": statements[0], "search_mode": "exact"})
    began = time.perf_counter()
    for _ in range(20):
        json.loads(payload)
    print("\njson decode of one insert statement : {:.3f} ms".format(
        (time.perf_counter() - began) / 20 * 1000
    ))
    began = time.perf_counter()
    for _ in range(20):
        parse(statements[0])
    print("parse of one insert statement       : {:.3f} ms".format(
        (time.perf_counter() - began) / 20 * 1000
    ))

    database.close()
    shutil.rmtree(directory, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
