#!/usr/bin/env python3
"""What is the peak memory of the container made of?

The peak memory is one of the three resources of the benchmark, and the judge
counts the page cache of the container as well.  This script runs the
workload without holding the data in memory and prints the memory of the
container after every phase.
"""
from __future__ import annotations

import json
import random
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, "/app")

from minivectordb.engine.database import Database  # noqa: E402
from minivectordb.engine.executor import Executor  # noqa: E402
from minivectordb.sqlparse import parse  # noqa: E402

ROWS = 10000
DIM = 128
BATCH = 100
QUERIES = 1000


def memory() -> dict:
    result = {}
    for name in ("memory.current", "memory.peak"):
        try:
            result[name] = int(Path("/sys/fs/cgroup/" + name).read_text().strip())
        except (OSError, ValueError):
            result[name] = 0
    try:
        result["anon"] = int(
            Path("/sys/fs/cgroup/memory.stat").read_text().split("anon ")[1].split("\n")[0]
        )
        result["file"] = int(
            Path("/sys/fs/cgroup/memory.stat").read_text().split("file ")[1].split("\n")[0]
        )
    except (OSError, ValueError, IndexError):
        result["anon"] = 0
        result["file"] = 0
    return result


def show(label: str) -> None:
    values = memory()
    print(
        "{:26s} current {:7.2f} MiB   peak {:7.2f} MiB   anon {:7.2f}   page cache {:7.2f}".format(
            label,
            values["memory.current"] / 1024 / 1024,
            values["memory.peak"] / 1024 / 1024,
            values["anon"] / 1024 / 1024,
            values["file"] / 1024 / 1024,
        )
    )


def statement(start: int, count: int) -> str:
    random.seed(20261111 + start)
    values = ",".join(
        "({},{},'[{}]')".format(
            index,
            index % 3,
            ",".join("%.6f" % random.uniform(-1.0, 1.0) for _ in range(DIM)),
        )
        for index in range(start, min(start + count, ROWS))
    )
    return "INSERT INTO docs VALUES " + values


def main() -> int:
    show("at the start")
    directory = Path("/data/memory")
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True, exist_ok=True)

    database = Database(directory)
    database.open()
    executor = Executor(database)
    executor.execute(parse("CREATE TABLE docs (id BIGINT PRIMARY KEY, g BIGINT, v VECTOR(128));"))
    show("after the table was created")

    began = time.perf_counter()
    for start in range(0, ROWS, BATCH):
        executor.execute(parse(statement(start, BATCH)))
    print("write {:.3f} s".format(time.perf_counter() - began))
    show("after the write")

    for number in range(20):
        random.seed(90000 + number)
        literal = "[" + ",".join("%.6f" % random.uniform(-1.0, 1.0) for _ in range(DIM)) + "]"
        if number % 2 == 0:
            sql = "SELECT id FROM docs WHERE g = {} ORDER BY L2_DISTANCE(v,'{}') LIMIT 10".format(
                number % 3, literal
            )
        else:
            sql = "SELECT id FROM docs ORDER BY L2_DISTANCE(v,'{}') LIMIT 10".format(literal)
        executor.execute(parse(sql))
    show("after 20 queries")

    began = time.perf_counter()
    for number in range(QUERIES):
        random.seed(90000 + number)
        literal = "[" + ",".join("%.6f" % random.uniform(-1.0, 1.0) for _ in range(DIM)) + "]"
        if number % 2 == 0:
            sql = "SELECT id FROM docs WHERE g = {} ORDER BY L2_DISTANCE(v,'{}') LIMIT 10".format(
                number % 3, literal
            )
        else:
            sql = "SELECT id FROM docs ORDER BY L2_DISTANCE(v,'{}') LIMIT 10".format(literal)
        executor.execute(parse(sql))
    print("query {:.3f} ms".format((time.perf_counter() - began) / QUERIES * 1000))
    show("after the queries")

    database.close()
    shutil.rmtree(directory, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
