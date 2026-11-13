#!/usr/bin/env python3
"""How does the IVF index behave on data that is not uniformly random?

Every recall number in this project was measured on vectors that are drawn
from one distribution over the whole cube.  Real data is usually clustered:
a few dense groups of vectors, and a query that lands inside one of them.
The coarse quantizer then has to place the query into a cluster that really
holds its neighbours, and the number of clusters that have to be probed grows.

This script builds three distributions and measures the recall of the
approximate search for each of them, with the same index and the same search
order as the engine:

* uniform: the vectors are uniform over [-1, 1]^128 (the frozen workload),
* clustered: 16 Gaussian blobs with a small spread, plus a uniform part,
* skewed: 80 percent of the vectors in one dense blob, the rest uniform.

The queries are drawn from the same distribution as the data, which is what a
real workload does.

Usage::

    python exp_ann_clustered.py [--rows 10000] [--queries 500]
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


def uniform(rng, rows, dim):
    return rng.uniform(-1.0, 1.0, size=(rows, dim)).astype(np.float32)


def clustered(rng, rows, dim, blobs=16, spread=0.08):
    part = rng.normal(size=(rows, dim)).astype(np.float32)
    part *= spread
    centres = rng.uniform(-0.7, 0.7, size=(blobs, dim)).astype(np.float32)
    which = rng.integers(0, blobs, size=rows)
    data = part + centres[which]
    return data.astype(np.float32)


def skewed(rng, rows, dim, share=0.8, spread=0.05):
    count = int(rows * share)
    blob = rng.normal(size=(count, dim)).astype(np.float32) * spread
    blob += rng.uniform(-0.3, 0.3, size=(1, dim)).astype(np.float32)
    rest = rng.uniform(-1.0, 1.0, size=(rows - count, dim)).astype(np.float32)
    return np.concatenate([blob, rest]).astype(np.float32)


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
    """The search of the engine."""
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
    began = time.perf_counter()
    for query in queries:
        exact = np.argsort(l2_distances(matrix, query), kind="stable")[:10]
        truth = {int(value) for value in exact}
        answered = set(search(centroids, members, radius, matrix, query, nprobe, 10))
        hits += len(answered & truth)
        total += len(truth)
    elapsed = time.perf_counter() - began
    return {
        "nprobe": nprobe,
        "recall": round(hits / total, 6) if total else 1.0,
        "ms_per_query": round(elapsed / len(queries) * 1000, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--queries", type=int, default=300)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261111)
    arguments = parser.parse_args()

    rng = np.random.default_rng(arguments.seed)
    rows = arguments.rows
    dim = arguments.dim

    distributions = {
        "uniform": uniform(rng, rows, dim),
        "clustered": clustered(rng, rows, dim),
        "skewed": skewed(rng, rows, dim),
    }

    print(
        "rows={} dim={} queries={} clusters={}".format(
            rows, dim, arguments.queries, CLUSTERS
        )
    )
    report = {}
    for name, matrix in distributions.items():
        if name == "uniform":
            queries = rng.uniform(-1.0, 1.0, size=(arguments.queries, dim)).astype(np.float32)
        elif name == "clustered":
            queries = clustered(rng, arguments.queries, dim)
        else:
            queries = skewed(rng, arguments.queries, dim)
        lines = []
        for nprobe in (16, 64):
            line = measure(matrix, queries, nprobe)
            lines.append(line)
            print(
                "  {:10s} nprobe={:3d}  recall={:.6f}  {:.3f} ms/query".format(
                    name, nprobe, line["recall"], line["ms_per_query"]
                )
            )
        report[name] = lines
    print("\nSUMMARY " + json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
