#!/usr/bin/env python3
"""What does the approximate mode really answer when the index is used?

Every "ann" measurement so far was taken on a table that is small enough for
the exact fallback, or on a table without an index at all, so the numbers
describe the exact scan and not the approximate search.  This script drives the
engine itself: it creates a vector index, puts the table above the exact
fallback threshold (MINIVECTORDB_ANN_EXACT_LIMIT, 50,000 rows by default) and
compares the answers of the approximate mode with the brute force answer.

It also measures the same thing for the value of nprobe that the engine uses
(clusters // 4 = 16 of 64 clusters) and for a complete scan of the index
(nprobe = 64), so that the cost of an exact answer through the index is
visible as well.

Usage::

    python exp_ann_real_engine.py [--rows 60000] [--queries 200]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, r"E:\MiniVectorDB\seekdb-round2")

# The fallback is switched off before the engine is imported, so that the
# approximate mode really uses the inverted lists of the index.
os.environ["MINIVECTORDB_ANN_EXACT_LIMIT"] = "0"

import numpy as np  # noqa: E402

from minivectordb.engine.database import Database  # noqa: E402
from minivectordb.engine.executor import Executor  # noqa: E402
from minivectordb.sqlparse import parse  # noqa: E402


def literal(vector) -> str:
    return "[" + ",".join("%.6f" % float(value) for value in vector) + "]"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=60000)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--batch", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261109)
    arguments = parser.parse_args()

    random.seed(arguments.seed)
    rng = np.random.default_rng(arguments.seed)
    directory = Path(tempfile.mkdtemp(prefix="mvdb-ann-real-"))
    database = Database(directory)
    database.open()
    executor = Executor(database)
    executor.execute(
        parse(
            "CREATE TABLE bench (id BIGINT PRIMARY KEY, category VARCHAR(8), e VECTOR(%d));"
            % arguments.dim
        )
    )
    vectors = rng.normal(size=(arguments.rows, arguments.dim)).astype(np.float32)
    categories = ["c%d" % index for index in range(10)]
    began = time.perf_counter()
    inserted = 0
    for start in range(0, arguments.rows, arguments.batch):
        rows = []
        for offset in range(start, min(start + arguments.batch, arguments.rows)):
            rows.append(
                "(%d, '%s', '%s')"
                % (offset + 1, categories[offset % len(categories)], literal(vectors[offset]))
            )
        answer = executor.execute(parse("INSERT INTO bench VALUES " + ", ".join(rows) + ";"))
        inserted += int(answer.get("affected_rows", 0))
    write_seconds = time.perf_counter() - began
    print("rows written: {} of {}".format(inserted, arguments.rows))
    began = time.perf_counter()
    executor.execute(parse("CREATE VECTOR INDEX idx_bench ON bench(e);"))
    build_seconds = time.perf_counter() - began
    print(
        "loaded {} rows x {} dims in {:.1f}s, index built in {:.1f}s".format(
            arguments.rows, arguments.dim, write_seconds, build_seconds
        )
    )

    query_vectors = rng.normal(size=(arguments.queries, arguments.dim)).astype(np.float32)
    category_of = np.asarray([index % len(categories) for index in range(arguments.rows)])
    category_number = np.asarray(
        [categories.index(categories[index % len(categories)]) for index in range(arguments.rows)]
    )

    # The brute force answer, computed over the whole table.
    truth = []
    for index in range(arguments.queries):
        query = query_vectors[index]
        difference = vectors - query
        distance = np.sqrt((difference * difference).sum(axis=1, dtype=np.float64))
        if index % 2 == 0:
            mask = category_number == index % len(categories)
            candidates = np.nonzero(mask)[0]
            picked = candidates[np.argsort(distance[candidates], kind="stable")[:10]]
        else:
            picked = np.argsort(distance, kind="stable")[:10]
        # The primary keys of the table start at 1.
        truth.append({int(value) + 1 for value in picked})

    def measure(mode, nprobe=None):
        database.mode = mode
        if nprobe is not None:
            for index in database.indexes.values():
                index.nprobe = nprobe
        hits = 0
        total = 0
        began = time.perf_counter()
        for index in range(arguments.queries):
            query = query_vectors[index]
            if index % 2 == 0:
                sql = (
                    "SELECT id FROM bench WHERE category = '%s' "
                    "ORDER BY L2_DISTANCE(e,'%s') LIMIT 10"
                ) % (categories[index % len(categories)], literal(query))
            else:
                sql = "SELECT id FROM bench ORDER BY L2_DISTANCE(e,'%s') LIMIT 10" % literal(query)
            answer = executor.execute(parse(sql))
            rows = answer.get("rows", [])
            answered = {int(row[0]) for row in rows}
            hits += len(answered & truth[index])
            total += min(10, len(truth[index]))
            if index == 0:
                print(
                    "  first query: answered {} rows, {} of them in the truth".format(
                        len(answered), len(answered & truth[index])
                    )
                )
        elapsed = time.perf_counter() - began
        return {
            "recall": round(hits / total, 6) if total else 1.0,
            "ms_per_query": round(elapsed / arguments.queries * 1000, 3),
        }

    # The fallback is switched off, so the approximate mode really uses the
    # inverted lists of the index.
    index = None
    for key, value in database.indexes.items():
        if key[1] == "idx_bench":
            index = value
    print("index clusters={} nprobe={}".format(index.clusters, index.nprobe))
    report = {
        "rows": arguments.rows,
        "queries": arguments.queries,
        "dim": arguments.dim,
        "ann_nprobe_default": measure("ann"),
        "ann_nprobe_64": measure("ann", nprobe=64),
        "exact": measure("exact"),
    }
    for name in ("ann_nprobe_default", "ann_nprobe_64", "exact"):
        line = report[name]
        print("  {:18s} recall={:.6f}  {:.3f} ms/query".format(name, line["recall"], line["ms_per_query"]))
    print("\nSUMMARY " + json.dumps(report))
    database.close()
    shutil.rmtree(directory, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
