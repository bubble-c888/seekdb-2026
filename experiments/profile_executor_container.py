#!/usr/bin/env python3
"""Profile the query path of the executor inside the evaluation container."""
import cProfile
import io
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

random.seed(20261111)
data_dir = Path("/data/profile_query")
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
pstats.Stats(profiler, stream=stream).sort_stats("tottime").print_stats(14)
print(stream.getvalue()[:3000])

database.close()
shutil.rmtree(data_dir, ignore_errors=True)
