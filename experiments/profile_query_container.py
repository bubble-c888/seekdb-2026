#!/usr/bin/env python3
"""Time the query path inside the evaluation container.

The script loads the frozen workload into a fresh data directory and then
times the queries in three ways: through the protocol, through the executor
and the distance computation alone.
"""
import random
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, "/app")

import numpy as np  # noqa: E402

from minivectordb.engine.database import Database  # noqa: E402
from minivectordb.engine.executor import Executor  # noqa: E402
from minivectordb.engine.vector import l2_distances  # noqa: E402
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
began = time.perf_counter()
for start in range(0, ROWS, 100):
    values = ",".join(
        "({},{},'[{}]')".format(
            index, groups[index], ",".join("%.6f" % value for value in vectors[index])
        )
        for index in range(start, min(start + 100, ROWS))
    )
    executor.execute(parse("INSERT INTO docs VALUES " + values + ";"))
print("write {:.2f}s".format(time.perf_counter() - began))

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

began = time.perf_counter()
for sql in queries:
    executor.execute(parse(sql))
print("query through the executor : {:.3f} ms".format((time.perf_counter() - began) / len(queries) * 1000))

began = time.perf_counter()
for sql in queries:
    parse(sql)
print("parse only                 : {:.3f} ms".format((time.perf_counter() - began) / len(queries) * 1000))

matrix = np.asarray(vectors, dtype=np.float32)
query = matrix[0]
began = time.perf_counter()
for _ in range(50):
    l2_distances(matrix, query)
print("l2_distances (10000 rows)  : {:.3f} ms".format((time.perf_counter() - began) / 50 * 1000))

began = time.perf_counter()
for _ in range(50):
    np.einsum("ij,ij->i", matrix, matrix)
print("einsum norms               : {:.3f} ms".format((time.perf_counter() - began) / 50 * 1000))

began = time.perf_counter()
for _ in range(50):
    matrix.dot(query)
print("dot                        : {:.3f} ms".format((time.perf_counter() - began) / 50 * 1000))

print("numpy", np.__version__, "threads", np.__config__.show() if hasattr(np, "__config__") else "")
database.close()
shutil.rmtree(data_dir, ignore_errors=True)
