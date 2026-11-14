#!/usr/bin/env python3
"""What does one query cost, part by part?

The query is the part of the benchmark with the biggest weight, so every part
of it is measured on its own: the distance computation, the order of the chunk
(the sort of 8192 rows), the primary keys and the merge into the running list.
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

sys.path.insert(0, "/app")

import numpy as np  # noqa: E402

from minivectordb.engine.vector import l2_distances  # noqa: E402

ROWS = 10000
CHUNK = 8192
DIM = 128
K = 10

random.seed(20261111)
matrix = np.asarray(
    [[random.uniform(-1.0, 1.0) for _ in range(DIM)] for _ in range(ROWS)], dtype=np.float32
)
pks = np.arange(ROWS, dtype=np.int64)
query = matrix[0].copy()


def timed(function, rounds=50):
    function()
    began = time.perf_counter()
    for _ in range(rounds):
        function()
    return (time.perf_counter() - began) / rounds * 1000


chunk = matrix[:CHUNK]
chunk_pks = pks[:CHUNK]
distances = l2_distances(chunk, query)

print("distance of a chunk of {:5d} rows : {:.3f} ms".format(CHUNK, timed(lambda: l2_distances(chunk, query))))
print("the same on the whole table     : {:.3f} ms".format(timed(lambda: l2_distances(matrix, query))))

print("lexsort of the chunk            : {:.3f} ms".format(
    timed(lambda: np.lexsort((chunk_pks, distances)))
))
print("argsort of the chunk            : {:.3f} ms".format(
    timed(lambda: np.argsort(distances, kind="stable"))
))


def partial():
    part = np.argpartition(distances, K - 1)[:K]
    return part[np.lexsort((chunk_pks[part], distances[part]))]


print("argpartition + a small lexsort  : {:.3f} ms".format(timed(partial)))
print("to list of the chunk            : {:.3f} ms".format(
    timed(lambda: distances.tolist())
))

# The merge into the running list of the ten best rows.
best = []


def merge():
    global best
    best = []
    keys = np.lexsort((chunk_pks, distances))
    for position in keys[:K].tolist():
        best.append((float(distances[position]), int(chunk_pks[position]), position))


print("the merge of the ten best rows   : {:.3f} ms".format(timed(merge)))
print("the whole query, roughly        : {:.3f} ms".format(
    timed(lambda: l2_distances(matrix, query))
    + timed(lambda: np.lexsort((chunk_pks, distances)))
    + timed(merge)
))
