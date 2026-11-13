#!/usr/bin/env python3
"""What does the executor really hand to the distance function?"""
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
data_dir = Path("/data/profile_chunk")
shutil.rmtree(data_dir, ignore_errors=True)
data_dir.mkdir(parents=True, exist_ok=True)

database = Database(data_dir)
database.open()
executor = Executor(database)
executor.execute(parse("CREATE TABLE docs (id BIGINT PRIMARY KEY, g BIGINT, v VECTOR(128));"))
vectors = [[random.uniform(-1.0, 1.0) for _ in range(DIM)] for _ in range(ROWS)]
for start in range(0, ROWS, 100):
    values = ",".join(
        "({},0,'[{}]')".format(index, ",".join("%.6f" % v for v in vectors[index]))
        for index in range(start, min(start + 100, ROWS))
    )
    executor.execute(parse("INSERT INTO docs VALUES " + values + ";"))

table = database.table("docs")
column = table.column("v")
query = np.asarray(vectors[0], dtype=np.float32)

chunk = table.read_vectors(column, 0, 8192)
print("chunk shape", chunk.shape, "dtype", chunk.dtype)
print("contiguous", chunk.flags["C_CONTIGUOUS"], "strides", chunk.strides, "owndata", chunk.flags["OWNDATA"])
print("base type", type(chunk.base))

for label, data in (
    ("the chunk", chunk),
    ("a copy of it", np.ascontiguousarray(chunk)),
    ("the whole column", table.read_vectors(column, 0, ROWS)),
):
    began = time.perf_counter()
    for _ in range(30):
        data.dot(query)
    print("dot on {:18s} : {:.3f} ms".format(label, (time.perf_counter() - began) / 30 * 1000))

database.close()
shutil.rmtree(data_dir, ignore_errors=True)
