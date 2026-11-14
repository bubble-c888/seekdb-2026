#!/usr/bin/env python3
"""The resource usage of the benchmark workload, as the judge measures it.

The judge counts the bytes of the data device (rbytes + wbytes) and the peak
memory of the whole container, the page cache included.  This script runs the
same workload and reports the numbers from the cgroup, so that the effect of a
change on the resources can be seen before a submission.

Run it in the container::

    docker run --rm --entrypoint python --network=none --cpus=2 --memory=512m \
        --read-only --cap-drop=ALL --security-opt=no-new-privileges \
        -e TMPDIR=/data -v /path/to/repo:/repo:ro -v mvdb-prof:/data \
        mvdb-change:local /repo/experiments/measure_resources.py
"""
from __future__ import annotations

import json
import os
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


def io_stat() -> dict:
    """The bytes of the data device, read from the cgroup."""
    result = {}
    for name in ("io.stat",):
        try:
            text = Path("/sys/fs/cgroup/" + name).read_text()
        except OSError:
            continue
        for line in text.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "rbytes":
                result["rbytes"] = int(parts[1])
            if len(parts) >= 2 and parts[0] == "wbytes":
                result["wbytes"] = int(parts[1])
    return result


def peak_memory() -> int:
    try:
        return int(Path("/sys/fs/cgroup/memory.peak").read_text().strip())
    except (OSError, ValueError):
        return 0


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> int:
    random.seed(20261111)
    vectors = [[random.uniform(-1.0, 1.0) for _ in range(DIM)] for _ in range(ROWS)]
    groups = [random.randrange(3) for _ in range(ROWS)]

    directory = Path("/data/resources")
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True, exist_ok=True)

    before = io_stat()
    database = Database(directory)
    database.open()
    executor = Executor(database)
    executor.execute(parse("CREATE TABLE docs (id BIGINT PRIMARY KEY, g BIGINT, v VECTOR(128));"))

    began = time.perf_counter()
    for start in range(0, ROWS, BATCH):
        values = ",".join(
            "({},{},'[{}]')".format(
                index,
                groups[index],
                ",".join("%.6f" % value for value in vectors[index]),
            )
            for index in range(start, min(start + BATCH, ROWS))
        )
        executor.execute(parse("INSERT INTO docs VALUES " + values + ";"))
    write_seconds = time.perf_counter() - began
    after_write = io_stat()
    print("write {:.3f} s   data dir {:.2f} MiB".format(
        write_seconds, directory_size(directory) / 1024 / 1024
    ))
    print("  written {:.2f} MiB   read {:.2f} MiB".format(
        (after_write.get("wbytes", 0) - before.get("wbytes", 0)) / 1024 / 1024,
        (after_write.get("rbytes", 0) - before.get("rbytes", 0)) / 1024 / 1024,
    ))
    print("  peak memory after the write: {:.2f} MiB".format(peak_memory() / 1024 / 1024))

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
    before_query = io_stat()
    began = time.perf_counter()
    for sql in queries:
        executor.execute(parse(sql))
    query_seconds = time.perf_counter() - began
    after_query = io_stat()
    print("query {:.3f} ms".format(query_seconds / len(queries) * 1000))
    print("  written {:.2f} MiB   read {:.2f} MiB".format(
        (after_query.get("wbytes", 0) - before_query.get("wbytes", 0)) / 1024 / 1024,
        (after_query.get("rbytes", 0) - before_query.get("rbytes", 0)) / 1024 / 1024,
    ))
    print("  peak memory: {:.2f} MiB".format(peak_memory() / 1024 / 1024))
    print("  data dir: {:.2f} MiB".format(directory_size(directory) / 1024 / 1024))

    database.close()
    shutil.rmtree(directory, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
