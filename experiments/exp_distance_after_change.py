#!/usr/bin/env python3
"""Does the new distance computation still answer like the old one?

The distance computation was rewritten to the expanded form.  The script
compares it with the previous implementation over the whole frozen workload:
all 1,000 queries, both the ten primary keys that are returned and the
distances themselves.  It also checks the case the expansion is weakest in:
a row whose vector is exactly the vector of the query, whose distance has to
be exactly zero.

Usage::

    python exp_distance_after_change.py [--rows 10000] [--queries 1000]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from minivectordb.engine.vector import l2_distances  # noqa: E402


_BUFFER = {}


def previous(matrix, query):
    """The implementation that was in the repository before the change.

    It builds the difference matrix in blocks inside a reusable buffer, so
    that a query does not allocate a temporary array every time.
    """
    query = query.astype(np.float32, copy=False)
    if matrix.dtype != np.float32:
        matrix = matrix.astype(np.float32)
    rows = matrix.shape[0]
    dim = matrix.shape[1]
    block = max(1, min(rows, (8 * 1024 * 1024) // max(dim * 4, 1)))
    squared = np.empty(rows, dtype=np.float64)
    buffer = _BUFFER.get(dim)
    if buffer is None or buffer.shape[0] < block:
        buffer = np.empty((block, dim), dtype=np.float32)
        _BUFFER[dim] = buffer
    for start in range(0, rows, block):
        part = matrix[start : start + block]
        diff = buffer[: part.shape[0]]
        np.subtract(part, query, out=diff)
        np.einsum(
            "ij,ij->i",
            diff,
            diff,
            dtype=np.float64,
            out=squared[start : start + part.shape[0]],
        )
    return np.sqrt(squared, out=squared)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261111)
    arguments = parser.parse_args()

    import random

    random.seed(arguments.seed)
    rows = arguments.rows
    dim = arguments.dim
    vectors = [[random.uniform(-1.0, 1.0) for _ in range(dim)] for _ in range(rows)]
    groups = [random.randrange(3) for _ in range(rows)]
    matrix = np.asarray(vectors, dtype=np.float32)
    ids = np.arange(rows, dtype=np.int64)

    differing = 0
    worst = 0.0
    worst_relative = 0.0
    examples = []
    for number in range(arguments.queries):
        vector = [random.uniform(-1.0, 1.0) for _ in range(dim)]
        query = np.asarray(vector, dtype=np.float32)
        if number % 2 == 0:
            group = random.randrange(3)
            mask = np.asarray([groups[i] == group for i in range(rows)])
            candidates = matrix[mask]
            candidate_ids = ids[mask]
        else:
            candidates = matrix
            candidate_ids = ids
        before = previous(candidates, query)
        after = l2_distances(candidates, query)
        worst = max(worst, float(np.max(np.abs(before - after))))
        worst_relative = max(
            worst_relative,
            float(np.max(np.abs(before - after) / np.maximum(before, 1e-12))),
        )
        expected = np.lexsort((candidate_ids, before))[:10]
        actual = np.lexsort((candidate_ids, after))[:10]
        if sorted(int(candidate_ids[i]) for i in expected) != sorted(
            int(candidate_ids[i]) for i in actual
        ):
            differing += 1
            if len(examples) < 3:
                examples.append({"query": number})

    print("queries                    : {}".format(arguments.queries))
    print("answers that differ        : {} / {}".format(differing, arguments.queries))
    print("worst distance difference  : {:.3e}".format(worst))
    print("worst relative difference  : {:.3e}".format(worst_relative))
    print("examples                   : {}".format(json.dumps(examples)))

    # A row that is exactly the query has to have the distance zero.
    exact = matrix[123]
    distance = l2_distances(np.asarray([exact]), exact)[0]
    print("\nexact match                : {!r} (previous {!r})".format(
        float(distance), float(previous(np.asarray([exact]), exact)[0])
    ))

    # The same row next to a row that is one step away.
    neighbour = exact.copy()
    neighbour[0] = np.float32(exact[0] + 1e-3)
    both = np.stack([exact, neighbour])
    distances = l2_distances(both, exact)
    print("exact match next to a near one: {!r} / {!r}".format(
        float(distances[0]), float(distances[1])
    ))

    query = np.asarray([random.uniform(-1.0, 1.0) for _ in range(dim)], dtype=np.float32)
    began = time.perf_counter()
    for _ in range(50):
        previous(matrix, query)
    old_ms = (time.perf_counter() - began) / 50 * 1000
    began = time.perf_counter()
    for _ in range(50):
        l2_distances(matrix, query)
    new_ms = (time.perf_counter() - began) / 50 * 1000
    print("\nprevious {:.3f} ms   new {:.3f} ms   ({:+.1f} %)".format(
        old_ms, new_ms, (new_ms - old_ms) / old_ms * 100
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
