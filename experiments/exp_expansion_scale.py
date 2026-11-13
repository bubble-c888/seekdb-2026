#!/usr/bin/env python3
"""How does the cancellation of the expansion depend on the dimension?

The expansion

    ||x - q||^2 = ||x||^2 - 2 x.q + ||q||^2

cancels: the norms are much larger than the squared distance, so the
subtraction loses digits.  The larger the dimension, the larger the norms
compared to the distance of two random points, so the cancellation should get
worse with the dimension and better with a smaller one.

This script measures the error of the expansion against the straightforward
computation for several shapes, and the distance of the tenth and the eleventh
nearest row, which is the margin that decides whether the order can change.

Usage::

    python exp_expansion_scale.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from minivectordb.engine.vector import l2_distances  # noqa: E402


def straightforward(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    diff = matrix.astype(np.float64) - query.astype(np.float64)
    return np.sqrt(np.einsum("ij,ij->i", diff, diff))


def measure(rows: int, dim: int, queries: int = 50, seed: int = 20261111) -> dict:
    rng = np.random.default_rng(seed)
    matrix = rng.uniform(-1.0, 1.0, size=(rows, dim)).astype(np.float32)
    worst = 0.0
    worst_relative = 0.0
    margins = []
    norms = []
    for _ in range(queries):
        query = rng.uniform(-1.0, 1.0, size=dim).astype(np.float32)
        expected = straightforward(matrix, query)
        actual = l2_distances(matrix, query)
        worst = max(worst, float(np.max(np.abs(expected - actual))))
        worst_relative = max(
            worst_relative,
            float(np.max(np.abs(expected - actual) / np.maximum(expected, 1e-12))),
        )
        ordered = np.sort(expected)
        if ordered.size > 10:
            margins.append(float(ordered[9] - ordered[8]))
        norms.append(float(np.mean(np.einsum("ij,ij->i", matrix, matrix, dtype=np.float64))))
    return {
        "rows": rows,
        "dim": dim,
        "mean_norm_squared": round(float(np.mean(norms)), 2),
        "max_abs_error": worst,
        "max_relative_error": worst_relative,
        "median_margin_between_10th_and_11th": float(np.median(margins)),
        "error_over_margin": worst / float(np.median(margins)),
    }


def main() -> int:
    shapes = [(10000, 128), (1000000, 16), (100000, 32), (200000, 64)]
    report = []
    for rows, dim in shapes:
        began = time.perf_counter()
        line = measure(rows, dim)
        line["seconds"] = round(time.perf_counter() - began, 1)
        report.append(line)
        print(
            "  {:9d} rows x {:3d} dims : |x|^2 ~ {:8.1f}   max abs error {:.3e}   "
            "relative {:.3e}   error/margin {:.3e}".format(
                rows,
                dim,
                line["mean_norm_squared"],
                line["max_abs_error"],
                line["max_relative_error"],
                line["error_over_margin"],
            )
        )
    print("\nSUMMARY " + json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
