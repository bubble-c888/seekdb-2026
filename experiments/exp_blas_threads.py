#!/usr/bin/env python3
"""What the thread pool of the linear algebra library costs.

The distance computation uses the expansion of the squared distance, so the
inner product runs through the matrix product of the library.  In the
evaluation container, which has two cores, that matrix product spawns a thread
pool: every call then pays for waking the threads up again.  The setting is
read when the library is loaded, so the script has to be started twice, once
with ``OPENBLAS_NUM_THREADS=1`` and once without.

Run it in the container::

    docker run --rm --entrypoint python --network=none --cpus=2 --memory=512m \
        --read-only --cap-drop=ALL --security-opt=no-new-privileges \
        -e TMPDIR=/data -v /path/to/repo:/repo:ro -v mvdb-prof:/data \
        mvdb-change:local /repo/experiments/exp_blas_threads.py

The script prints the thread setting it was started with, so that the two
runs can be compared.
"""
from __future__ import annotations

import cProfile
import io
import os
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

print("OPENBLAS_NUM_THREADS =", os.environ.get("OPENBLAS_NUM_THREADS", "(not set)"))

random.seed(20261111)
data_dir = Path("/data/blas_threads")
shutil.rmtree(data_dir, ignore_errors=True)
data_dir.mkdir(parents=True, exist_ok=True)

database = Database(data_dir)
database.open()
database.mode = "exact"
executor = Executor(database)
executor.execute(parse("CREATE TABLE docs (id BIGINT PRIMARY KEY, g BIGINT, v VECTOR(128));"))
vectors = [[random.uniform(-1.0, 1.0) for _ in range(DIM)] for _ in range(ROWS)]
groups = [random.randrange(3) for _ in range(ROWS)]
for start in range(0, ROWS, 100):
    values = ",".join(
        "({},{},'[{}]')".format(
            index, groups[index], ",".join("%.6f" % value for value in vectors[index])
        )
        for index in range(start, min(start + 100, ROWS))
    )
    executor.execute(parse("INSERT INTO docs VALUES " + values + ";"))

queries = []
for number in range(200):
    vector = [random.uniform(-1.0, 1.0) for _ in range(DIM)]
    literal = "[" + ",".join("%.6f" % value for value in vector) + "]"
    if number % 2 == 0:
        sql = "SELECT id FROM docs WHERE g = {} ORDER BY L2_DISTANCE(v,'{}') LIMIT 10".format(
            random.randrange(3), literal
        )
    else:
        sql = "SELECT id FROM docs ORDER BY L2_DISTANCE(v,'{}') LIMIT 10".format(literal)
    queries.append(sql)

executor.execute(parse(queries[0]))
began = time.perf_counter()
for sql in queries:
    executor.execute(parse(sql))
print("query through the executor : {:.3f} ms".format((time.perf_counter() - began) / len(queries) * 1000))

profiler = cProfile.Profile()
profiler.enable()
for sql in queries[:60]:
    executor.execute(parse(sql))
profiler.disable()
stream = io.StringIO()
stats = pstats.Stats(profiler, stream=stream).sort_stats("tottime")
stats.print_stats(4)
for line in stream.getvalue().splitlines():
    if "dot" in line or "einsum" in line or "ncalls" in line:
        print(line)

database.close()
shutil.rmtree(data_dir, ignore_errors=True)
