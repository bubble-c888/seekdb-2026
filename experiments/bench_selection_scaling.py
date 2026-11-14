#!/usr/bin/env python3
"""When does the selection of the best rows stop paying off?

The query selects the rows that can enter the running list with a partition
and sorts only those.  The claim is that this does not work for a large K:
the more rows are wanted, the closer the candidate set is to the whole chunk
and the less the complete sort costs.

This script measures the complete sort of a chunk against the selection for
several values of K.
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

sys.path.insert(0, "/app")

import numpy as np  # noqa: E402

CHUNK = 8192
DIM = 128


def timed(function, rounds=30):
    function()
    began = time.perf_counter()
    for _ in range(rounds):
        function()
    return (time.perf_counter() - began) / rounds * 1000


def main() -> int:
    random.seed(20261111)
    matrix = np.asarray(
        [[random.uniform(-1.0, 1.0) for _ in range(DIM)] for _ in range(CHUNK)], dtype=np.float32
    )
    pks = np.arange(CHUNK, dtype=np.int64)
    query = matrix[0].copy()
    difference = matrix - query
    distances = np.sqrt((difference * difference).sum(axis=1, dtype=np.float64))

    print("chunk of {} rows".format(CHUNK))
    for wanted in (10, 100, 1000, 4096, 8192):
        complete = timed(lambda: np.lexsort((pks, distances)))

        def selection():
            part = np.argpartition(distances, wanted - 1)[:wanted]
            cut = float(distances[part].max())
            keep = np.zeros(distances.size, dtype=bool)
            keep[part] = True
            keep |= distances == cut
            candidates = np.nonzero(keep)[0]
            return candidates[np.lexsort((pks[candidates], distances[candidates]))]

        chosen = timed(selection)
        print(
            "  K={:5d}  complete sort {:.3f} ms   selection {:.3f} ms   factor {:.2f}".format(
                wanted, complete, chosen, complete / chosen if chosen else 0.0
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
