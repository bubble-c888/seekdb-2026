#!/usr/bin/env python3
"""Is the expansion slow on a view of a bigger array?

The executor hands the distance function a chunk of the cached column, which
is a view into a bigger array.  The matrix product of the library may take a
different path for such an input.
"""
import random
import sys
import time

sys.path.insert(0, "/app")

import numpy as np  # noqa: E402

from minivectordb.engine.vector import l2_distances  # noqa: E402

ROWS = 10000
DIM = 128
random.seed(20261111)
vectors = [[random.uniform(-1.0, 1.0) for _ in range(DIM)] for _ in range(ROWS)]
matrix = np.asarray(vectors, dtype=np.float32)
query = matrix[0].copy()

big = np.zeros((ROWS * 2, DIM), dtype=np.float32)
big[:ROWS] = matrix
view = big[0:ROWS]
contiguous = np.ascontiguousarray(matrix)

for label, data in (("contiguous", contiguous), ("view", view)):
    began = time.perf_counter()
    for _ in range(50):
        l2_distances(data, query)
    print("l2_distances on a {} array : {:.3f} ms".format(
        label, (time.perf_counter() - began) / 50 * 1000
    ))
    began = time.perf_counter()
    for _ in range(50):
        data.dot(query)
    print("dot on a {} array          : {:.3f} ms".format(
        label, (time.perf_counter() - began) / 50 * 1000
    ))
    began = time.perf_counter()
    for _ in range(50):
        np.einsum("ij,ij->i", data, data)
    print("einsum on a {} array      : {:.3f} ms".format(
        label, (time.perf_counter() - began) / 50 * 1000
    ))
    began = time.perf_counter()
    for _ in range(50):
        np.einsum("ij,j->i", data, query, dtype=np.float64)
    print("einsum dot on a {} array  : {:.3f} ms".format(
        label, (time.perf_counter() - began) / 50 * 1000
    ))
