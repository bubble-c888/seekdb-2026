#!/usr/bin/env python3
"""Does a faster distance implementation return the same TopK as the current one?

The earlier check compared 200 random queries and found that none of the three
faster spellings was bit identical to the shipped computation.  A difference of
1e-7 does not matter on its own, but the distances decide the order and the task
package breaks ties by the primary key, so a difference can still change which
rows are returned.

This script therefore does not compare distances: it compares the *answers*.
It rebuilds the frozen benchmark workload (10,000 rows of 128 dimensions, the
same seed as scripts/perf_container.py, 1,000 TopK queries with K = 10 and half
of them filtered by an equality on a normal column), and for every single query
it compares the set of returned primary keys of the shipped implementation with
the sets returned by the faster spellings.

Usage::

    python exp_distance_equivalence.py [--rows 10000] [--queries 1000]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time

sys.path.insert(0, r"E:\MiniVectorDB\seekdb-round2")

import numpy as np  # noqa: E402

from minivectordb.engine.vector import l2_distances  # noqa: E402

ROWS = 10000
DIM = 128
QUERIES = 1000
SEED = 20261111


def shipped(matrix, query):
    """The distance computation that is in the submitted version."""
    return l2_distances(matrix, query)


def with_multiply(matrix, query):
    """float32 difference, float32 product, float64 accumulation."""
    diff = matrix - query
    squared = np.empty(matrix.shape[0], dtype=np.float64)
    np.sum(diff * diff, axis=1, dtype=np.float64, out=squared)
    return np.sqrt(squared, out=squared)


def with_blas(matrix, query):
    """The expansion of the squared distance with a matrix product."""
    dots = matrix.dot(query)
    norms = np.einsum("ij,ij->i", matrix, matrix, dtype=np.float64)
    squared = norms - 2.0 * dots + float(np.dot(query, query))
    np.maximum(squared, 0.0, out=squared)
    return np.sqrt(squared)


def pick(distances, ids, k=10):
    """The TopK of the engine: order by (distance, primary key)."""
    order = np.lexsort((ids, distances))[:k]
    return order, distances[order]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=ROWS)
    parser.add_argument("--queries", type=int, default=QUERIES)
    parser.add_argument("--dim", type=int, default=DIM)
    parser.add_argument("--seed", type=int, default=SEED)
    arguments = parser.parse_args()

    random.seed(arguments.seed)
    rows = arguments.rows
    dim = arguments.dim

    vectors = [[random.uniform(-1.0, 1.0) for _ in range(dim)] for _ in range(rows)]
    groups = [random.randrange(3) for _ in range(rows)]
    matrix = np.asarray(vectors, dtype=np.float32)
    ids = np.arange(rows, dtype=np.int64)

    print("workload: {} rows x {} dims, {} queries, K=10".format(rows, dim, arguments.queries))
    print("matrix bytes: {:.1f} MiB".format(matrix.nbytes / 1024 / 1024))

    variants = (
        ("multiply+sum", with_multiply),
        ("blas expansion", with_blas),
    )

    # The query set is built once, so that every spelling of the distance sees
    # exactly the same workload.
    workload = []
    for number in range(arguments.queries):
        vector = [random.uniform(-1.0, 1.0) for _ in range(dim)]
        group = None
        if number % 2 == 0:
            group = random.randrange(3)
            mask = np.asarray([groups[i] == group for i in range(rows)])
            candidates = matrix[mask]
            candidates_ids = ids[mask]
        else:
            candidates = matrix
            candidates_ids = ids
        workload.append((number, np.asarray(vector, dtype=np.float32), candidates, candidates_ids))

    # The boundary gap tells how likely a tie is to be broken the other way.
    gaps = []
    for number, query, candidates, candidates_ids in workload:
        distances = shipped(candidates, query)
        order, ordered = pick(distances, candidates_ids)
        if order.size >= 10:
            gaps.append(float(ordered[9] - ordered[8]))
    gaps = np.asarray(gaps)
    print("\n== how close are the distances at the K boundary (row 9 vs row 10) ==")
    print("  smallest gap  : {:.3e}".format(float(gaps.min())))
    print("  1st percentile: {:.3e}".format(float(np.percentile(gaps, 1))))
    print("  median gap    : {:.3e}".format(float(np.median(gaps))))
    print("  gaps below 1e-6: {} of {}".format(int((gaps < 1e-6).sum()), gaps.size))

    summary = {}
    for name, function in variants:
        differing = 0
        differing_examples = []
        worst = 0.0
        for number, query, candidates, candidates_ids in workload:
            expected = shipped(candidates, query)
            actual = function(candidates, query)
            worst = max(worst, float(np.max(np.abs(expected - actual))))
            expected_order, _ = pick(expected, candidates_ids)
            actual_order, _ = pick(actual, candidates_ids)
            expected_ids = sorted(int(candidates_ids[i]) for i in expected_order)
            actual_ids = sorted(int(candidates_ids[i]) for i in actual_order)
            if expected_ids != actual_ids:
                differing += 1
                if len(differing_examples) < 3:
                    missing = sorted(set(expected_ids) - set(actual_ids))
                    extra = sorted(set(actual_ids) - set(expected_ids))
                    differing_examples.append(
                        {"query": number, "missing": missing, "extra": extra}
                    )
        summary[name] = {
            "queries": arguments.queries,
            "answers_differing": differing,
            "worst_distance_difference": worst,
            "examples": differing_examples,
        }
        print("\n== {} ==".format(name))
        print("  queries answered differently : {} / {}".format(differing, arguments.queries))
        print("  worst distance difference    : {:.3e}".format(worst))
        for example in differing_examples:
            print("  example: {}".format(json.dumps(example)))

    # The timing, measured on the same data.
    print("\n== time per query, 10,000 x 128 ==")
    query = np.asarray([random.uniform(-1.0, 1.0) for _ in range(dim)], dtype=np.float32)
    for name, function in (("shipped", shipped),) + variants:
        function(matrix, query)
        began = time.perf_counter()
        for _ in range(50):
            function(matrix, query)
        print("  {:16s} {:.3f} ms".format(name, (time.perf_counter() - began) / 50 * 1000))

    print("\nSUMMARY " + json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
