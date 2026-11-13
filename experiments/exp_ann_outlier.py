#!/usr/bin/env python3
"""Is the approximate search worse for a query that is an outlier?

The conclusion that the clustered data is the easy case for the inverted
lists was measured with queries that are drawn from the same distribution as
the data.  A query that lies outside the cloud of the data is the obvious
counter example: its nearest neighbours are spread over the clusters, so
probing a quarter of them does not find all of them.

This script measures the recall of the approximate search for three kinds of
queries against the same table of uniformly random vectors:

* in the distribution: the query is drawn like the data,
* near the edge: the query is pushed towards the border of the cube,
* an outlier: the query is far outside the cube, so its neighbours are spread
  over all the clusters.

Usage::

    python exp_ann_outlier.py [--rows 10000] [--queries 300]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from minivectordb.engine.vector import kmeans, l2_distances  # noqa: E402

CLUSTERS = 64
KMEANS_SEED = 20261109


def build_index(matrix):
    centroids = kmeans(matrix, CLUSTERS, seed=KMEANS_SEED)
    assignment = np.argmin(
        np.stack([l2_distances(matrix, centroid) for centroid in centroids], axis=1), axis=1
    )
    members = []
    radius = np.zeros(CLUSTERS, dtype=np.float64)
    for index in range(CLUSTERS):
        slots = np.nonzero(assignment == index)[0]
        members.append(slots)
        if slots.size:
            radius[index] = float(l2_distances(matrix[slots], centroids[index]).max())
    return centroids, members, radius


def search(centroids, members, radius, matrix, query, nprobe, wanted):
    distances = l2_distances(centroids, query)
    best = []
    seen = set()
    probed = 0
    for cluster in np.argsort(distances, kind="stable"):
        if probed >= 1 and len(best) >= wanted:
            bound = max(0.0, float(distances[cluster]) - float(radius[cluster]))
            if bound > abs(best[-1][0]):
                break
        if probed >= nprobe and len(best) >= wanted:
            break
        probed += 1
        for slot in members[int(cluster)]:
            slot = int(slot)
            if slot in seen:
                continue
            seen.add(slot)
            distance = float(l2_distances(matrix[slot : slot + 1], query)[0])
            candidate = (distance, slot)
            if len(best) < wanted:
                best.append(candidate)
                best.sort(key=lambda item: (item[0], item[1]))
            elif (candidate[0], candidate[1]) < (best[-1][0], best[-1][1]):
                best[-1] = candidate
                best.sort(key=lambda item: (item[0], item[1]))
    return [slot for _, slot in best]


def measure(matrix, queries, nprobe):
    centroids, members, radius = build_index(matrix)
    hits = 0
    total = 0
    for query in queries:
        exact = np.argsort(l2_distances(matrix, query), kind="stable")[:10]
        truth = {int(value) for value in exact}
        answered = set(search(centroids, members, radius, matrix, query, nprobe, 10))
        hits += len(answered & truth)
        total += len(truth)
    return hits / total if total else 1.0


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
    matrix = rng.uniform(-1.0, 1.0, size=(rows, dim)).astype(np.float32)

    kinds = {
        "in the distribution": rng.uniform(-1.0, 1.0, size=(arguments.queries, dim)).astype(
            np.float32
        ),
        "near the edge": (
            rng.uniform(-1.0, 1.0, size=(arguments.queries, dim))
            + rng.choice([-1.0, 1.0], size=(arguments.queries, 1))
            * 1.5
        ).astype(np.float32),
        "an outlier": (
            rng.uniform(-1.0, 1.0, size=(arguments.queries, dim))
            + rng.choice([-1.0, 1.0], size=(arguments.queries, 1))
            * 6.0
        ).astype(np.float32),
    }

    print(
        "rows={} dim={} queries={} clusters={}".format(rows, dim, arguments.queries, CLUSTERS)
    )
    report = {}
    for name, queries in kinds.items():
        lines = []
        for nprobe in (16, 64):
            began = time.perf_counter()
            recall = measure(matrix, queries, nprobe)
            elapsed = (time.perf_counter() - began) / len(queries) * 1000
            lines.append({"nprobe": nprobe, "recall": round(recall, 6), "ms_per_query": round(elapsed, 3)})
            print(
                "  {:20s} nprobe={:3d}  recall={:.6f}  {:.2f} ms/query".format(
                    name, nprobe, recall, elapsed
                )
            )
        report[name] = lines
    print("\nSUMMARY " + json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
