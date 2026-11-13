#!/usr/bin/env python3
"""How much of the ANN recall does nprobe actually buy?

The index uses 64 coarse clusters and probes 16 of them, which is a quarter of
the index.  The measured recall of the submitted version is 1.0, so the question
is whether that is bought by the setting or by the data: if a smaller nprobe
keeps the recall at 1.0, the approximate mode could answer faster, and if it
does not, the setting has to stay where it is because the hidden cases compare
the approximate answer with the exact one.

The script rebuilds the frozen workload (10,000 rows of 128 dimensions, the
seed of scripts/perf_container.py, 1,000 queries with K = 10 and half of them
filtered), builds the same coarse quantizer as the engine (64 clusters, seed
20261109) and replays the search of the engine, including its early stop on
the cluster bound, for every nprobe in the list.

Usage::

    python exp_ann_nprobe.py [--rows 10000] [--queries 1000]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time

sys.path.insert(0, r"E:\MiniVectorDB\seekdb-round2")

import numpy as np  # noqa: E402

from minivectordb.engine.vector import kmeans, l2_distances  # noqa: E402

ROWS = 10000
DIM = 128
QUERIES = 1000
SEED = 20261111
CLUSTERS = 64
KMEANS_SEED = 20261109


def build_index(matrix, clusters):
    """The coarse quantizer of the engine, plus the radius of every cluster."""
    centroids = kmeans(matrix, clusters, seed=KMEANS_SEED)
    assignment = np.argmin(
        np.stack([l2_distances(matrix, centroid) for centroid in centroids], axis=1), axis=1
    )
    members = []
    radius = np.zeros(clusters, dtype=np.float64)
    for index in range(clusters):
        slots = np.nonzero(assignment == index)[0]
        members.append(slots)
        if slots.size:
            distances = l2_distances(matrix[slots], centroids[index])
            radius[index] = float(distances.max())
    return centroids, members, radius


def search(centroids, members, radius, query, nprobe, wanted, filter_mask=None):
    """The search of the engine: clusters by distance, early stop on the bound."""
    order = np.argsort(l2_distances(centroids, query), kind="stable")
    best = []
    probed = 0
    seen = set()
    for cluster in order:
        if probed >= 1 and len(best) >= wanted:
            bound = max(0.0, float(l2_distances(centroids, query)[cluster]) - float(radius[cluster]))
            if bound > abs(best[-1][0]):
                break
        if probed >= nprobe and len(best) >= wanted:
            break
        probed += 1
        slots = members[int(cluster)]
        if filter_mask is not None:
            slots = slots[filter_mask[slots]]
        for slot in slots:
            slot = int(slot)
            if slot in seen:
                continue
            seen.add(slot)
            distance = float(l2_distances(matrix_of(slot), query)[0])
            candidate = (distance, slot)
            if len(best) < wanted:
                best.append(candidate)
                best.sort(key=lambda item: (item[0], item[1]))
            elif (candidate[0], candidate[1]) < (best[-1][0], best[-1][1]):
                best[-1] = candidate
                best.sort(key=lambda item: (item[0], item[1]))
    return [slot for _, slot in best]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=ROWS)
    parser.add_argument("--queries", type=int, default=QUERIES)
    parser.add_argument("--dim", type=int, default=DIM)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--nprobe", default="1,2,4,8,16,32,64", help="comma separated list of nprobe values"
    )
    arguments = parser.parse_args()

    global matrix_of  # noqa: PLW0603

    random.seed(arguments.seed)
    rows = arguments.rows
    dim = arguments.dim

    vectors = [[random.uniform(-1.0, 1.0) for _ in range(dim)] for _ in range(rows)]
    groups = [random.randrange(3) for _ in range(rows)]
    matrix = np.asarray(vectors, dtype=np.float32)
    ids = np.arange(rows, dtype=np.int64)
    matrix_of = lambda slot: matrix[slot : slot + 1]  # noqa: E731

    workload = []
    for number in range(arguments.queries):
        vector = [random.uniform(-1.0, 1.0) for _ in range(dim)]
        query = np.asarray(vector, dtype=np.float32)
        group = None
        if number % 2 == 0:
            group = random.randrange(3)
            mask = np.asarray([groups[i] == group for i in range(rows)])
        else:
            mask = None
        workload.append((number, query, group, mask))

    print("workload: {} rows x {} dims, {} queries, K=10".format(rows, dim, arguments.queries))
    began = time.perf_counter()
    centroids, members, radius = build_index(matrix, CLUSTERS)
    print("index: {} clusters, built in {:.1f}s".format(CLUSTERS, time.perf_counter() - began))
    sizes = [len(slot) for slot in members]
    print("cluster sizes: min={} max={} mean={:.1f}".format(min(sizes), max(sizes), sum(sizes) / len(sizes)))

    # The ground truth: the exact answer, without any index.
    truth = []
    for number, query, group, mask in workload:
        candidates = matrix if mask is None else matrix[mask]
        candidate_ids = ids if mask is None else ids[mask]
        distances = l2_distances(candidates, query)
        order = np.lexsort((candidate_ids, distances))[:10]
        truth.append({int(candidate_ids[i]) for i in order})

    report = []
    for nprobe in [int(value) for value in arguments.nprobe.split(",")]:
        hits = 0
        total = 0
        began = time.perf_counter()
        for (number, query, group, mask), expected in zip(workload, truth):
            answered = set(search(centroids, members, radius, query, nprobe, 10, mask))
            hits += len(answered & expected)
            total += len(expected)
        elapsed = time.perf_counter() - began
        recall = hits / total if total else 1.0
        line = {
            "nprobe": nprobe,
            "recall": round(recall, 6),
            "ms_per_query": round(elapsed / arguments.queries * 1000, 3),
        }
        report.append(line)
        print("  nprobe={:3d}  recall={:.6f}  {:.3f} ms/query".format(nprobe, recall, elapsed / arguments.queries * 1000))

    print("\nSUMMARY " + json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
