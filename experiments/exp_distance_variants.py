#!/usr/bin/env python3
"""Which spelling of the expanded squared distance is safe and fast?

The decision is to compute the squared distance as

    ||x - q||^2 = ||x||^2 - 2 x.q + ||q||^2

so that the inner product can run through the matrix product of the library.
The expansion cancels: when the distance is small compared to the norms, the
result loses digits.  This script measures the error of every spelling against
the straightforward computation and its cost, on uniform and on clustered data.

Usage::

    python exp_distance_variants.py [--rows 10000] [--queries 200]
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

EPS = float(np.finfo(np.float64).eps)


def expansion_einsum64(matrix, query):
    """Norms and inner product accumulated in float64."""
    norms = np.einsum("ij,ij->i", matrix, matrix, dtype=np.float64)
    dots = np.einsum("ij,j->i", matrix, query, dtype=np.float64)
    squared = norms - 2.0 * dots + float(np.einsum("j,j->", query, query, dtype=np.float64))
    np.maximum(squared, 0.0, out=squared)
    return squared


def expansion_blas32(matrix, query):
    """The inner product through the float32 matrix product of the library."""
    norms = np.einsum("ij,ij->i", matrix, matrix, dtype=np.float64)
    dots = matrix.dot(query)
    squared = norms - 2.0 * dots.astype(np.float64) + float(np.dot(query, query))
    np.maximum(squared, 0.0, out=squared)
    return squared


def expansion_blas64(matrix, query):
    """The inner product through a float64 matrix product."""
    norms = np.einsum("ij,ij->i", matrix, matrix, dtype=np.float64)
    dots = matrix.astype(np.float64).dot(query.astype(np.float64))
    squared = norms - 2.0 * dots + float(np.dot(query.astype(np.float64), query.astype(np.float64)))
    np.maximum(squared, 0.0, out=squared)
    return squared


def repair(matrix, query, squared):
    """Measure the rows whose digits the expansion may have destroyed."""
    norms = np.einsum("ij,ij->i", matrix, matrix, dtype=np.float64)
    bound = 64.0 * EPS * (norms + float(np.einsum("j,j->", query, query, dtype=np.float64)))
    risky = np.nonzero(squared <= bound)[0]
    for index in risky:
        diff = matrix[index].astype(np.float64) - query.astype(np.float64)
        squared[index] = float(np.dot(diff, diff))
    return len(risky)


def timed(function, matrix, query, rounds=30):
    function(matrix, query)
    began = time.perf_counter()
    for _ in range(rounds):
        function(matrix, query)
    return (time.perf_counter() - began) / rounds * 1000


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261111)
    arguments = parser.parse_args()

    rng = np.random.default_rng(arguments.seed)
    rows = arguments.rows
    dim = arguments.dim
    uniform = rng.uniform(-1.0, 1.0, size=(rows, dim)).astype(np.float32)
    clustered = (
        rng.normal(size=(rows, dim)).astype(np.float32) * 0.05
        + rng.uniform(-0.5, 0.5, size=(16, dim)).astype(np.float32)[
            rng.integers(0, 16, size=rows)
        ]
    ).astype(np.float32)

    variants = {
        "expansion einsum64": expansion_einsum64,
        "expansion blas32": expansion_blas32,
        "expansion blas64": expansion_blas64,
    }

    for name, matrix in (("uniform", uniform), ("clustered", clustered)):
        queries = rng.uniform(-1.0, 1.0, size=(arguments.queries, dim)).astype(np.float32)
        print("\n== {} ==".format(name))
        reference = l2_distances(matrix, queries[0])
        for label, function in variants.items():
            worst = 0.0
            worst_distance = 0.0
            risky_total = 0
            for query in queries:
                expected = l2_distances(matrix, query)
                actual = np.sqrt(function(matrix, query))
                worst = max(worst, float(np.max(np.abs(expected - actual))))
                worst_distance = max(
                    worst_distance,
                    float(
                        np.max(
                            np.abs(expected - actual)
                            / np.maximum(expected, 1e-12)
                        )
                    ),
                )
                risky_total += repair(matrix, query, function(matrix, query))
            print(
                "  {:22s} max abs error {:.3e}   max rel error {:.3e}   rows to repair {:6d}".format(
                    label, worst, worst_distance, risky_total // max(1, len(queries))
                )
            )
        query = queries[0]
        print("  {:22s} {:.3f} ms".format("shipped", timed(l2_distances, matrix, query)))
        for label, function in variants.items():
            print(
                "  {:22s} {:.3f} ms".format(
                    label, timed(lambda m, q, f=function: f(m, q), matrix, query)
                )
            )

    print("\nSUMMARY done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
