"""Performance benchmark for MiniVectorDB.

The workload follows the description of the competition benchmark: a small
single client load with 10,000 rows of 128 dimensional vectors that are
written in batches of 100 rows, followed by 1,000 vector TopK queries with
K = 10, half of them with an equality filter on a normal column.

The script measures

* the wall clock time of the write phase and of the query phase,
* the peak memory of the process,
* Recall@10 of the filtered and the unfiltered queries.

Usage::

    python tools/benchmark.py [--rows 10000] [--queries 1000] [--dim 128] [--mode exact]
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from minivectordb.engine.database import Database  # noqa: E402
from minivectordb.engine.executor import Executor  # noqa: E402
from minivectordb.engine.vector import l2_distances  # noqa: E402
from minivectordb.sqlparse import parse  # noqa: E402


def format_vector(values: np.ndarray) -> str:
    return "[" + ",".join(f"{float(v):.6g}" for v in values) + "]"


def peak_memory_mib() -> float:
    """Peak resident memory of this process in MiB.

    Uses ``resource.getrusage`` on Unix and ``GetProcessMemoryInfo`` on
    Windows, so the number is meaningful on every platform.
    """
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports kilobytes, macOS reports bytes.
        return round(usage / 1024.0, 2) if usage > 1024 * 1024 else round(usage / 1024.0, 2)
    except Exception:
        pass
    try:
        import ctypes
        from ctypes import wintypes

        class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = _PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        psapi = ctypes.windll.psapi
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_PROCESS_MEMORY_COUNTERS),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        handle = kernel32.GetCurrentProcess()
        if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return round(counters.PeakWorkingSetSize / (1024.0 * 1024.0), 2)
    except Exception:
        pass
    return 0.0


def run(args) -> dict:
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    data_dir = Path(args.data_dir or tempfile.mkdtemp(prefix="minivectordb-bench-"))
    database = Database(data_dir)
    database.open()
    database.mode = args.mode
    executor = Executor(database)

    categories = [f"c{index}" for index in range(10)]

    # ---------------------------------------------------------------- write
    vectors = rng.normal(size=(args.rows, args.dim)).astype(np.float32)
    write_start = time.perf_counter()
    executor.execute(parse("CREATE TABLE bench (id BIGINT PRIMARY KEY, category VARCHAR(8), e VECTOR(%d));" % args.dim))
    batch = args.batch
    for start in range(0, args.rows, batch):
        rows = []
        for offset in range(start, min(start + batch, args.rows)):
            rows.append(
                "(%d, '%s', '%s')"
                % (
                    offset + 1,
                    categories[offset % len(categories)],
                    format_vector(vectors[offset]),
                )
            )
        executor.execute(parse("INSERT INTO bench VALUES " + ", ".join(rows) + ";"))
    write_time = time.perf_counter() - write_start

    # ---------------------------------------------------------------- query
    query_vectors = rng.normal(size=(args.queries, args.dim)).astype(np.float32)
    category_of = np.asarray(
        [index % len(categories) for index in range(args.rows)], dtype=np.int64
    )
    recall_filtered = []
    recall_unfiltered = []
    query_time = 0.0
    truth_time = 0.0
    for index in range(args.queries):
        query = query_vectors[index]
        filtered = index % 2 == 0
        if filtered:
            category = categories[index % len(categories)]
            sql = (
                "SELECT id, L2_DISTANCE(e, '%s') AS distance FROM bench "
                "WHERE category = '%s' ORDER BY distance LIMIT %d;"
                % (format_vector(query), category, args.k)
            )
        else:
            sql = (
                "SELECT id, L2_DISTANCE(e, '%s') AS distance FROM bench "
                "ORDER BY distance LIMIT %d;" % (format_vector(query), args.k)
            )
        statement = parse(sql)
        start = time.perf_counter()
        result = executor.execute(statement)
        query_time += time.perf_counter() - start
        returned = [row[0] for row in result["rows"]]
        # The ground truth is computed outside of the measured section.
        truth_start = time.perf_counter()
        mask = (
            category_of == (index % len(categories))
            if filtered
            else np.ones(args.rows, dtype=np.bool_)
        )
        candidates = vectors[mask]
        distances = l2_distances(candidates, query.astype(np.float32))
        order = np.argsort(distances, kind="stable")[: args.k]
        truth = [int(offset + 1) for offset in np.nonzero(mask)[0][order]]
        hits = len(set(returned) & set(truth))
        recall = hits / max(1, min(args.k, len(truth)))
        if filtered:
            recall_filtered.append(recall)
        else:
            recall_unfiltered.append(recall)
        truth_time += time.perf_counter() - truth_start

    peak = peak_memory_mib()
    database.close()
    if args.data_dir is None:
        shutil.rmtree(data_dir, ignore_errors=True)

    return {
        "rows": args.rows,
        "queries": args.queries,
        "dim": args.dim,
        "k": args.k,
        "mode": args.mode,
        "write_seconds": round(write_time, 3),
        "query_seconds": round(query_time, 3),
        "query_ms_each": round(query_time / max(1, args.queries) * 1000, 3),
        "ground_truth_seconds": round(truth_time, 3),
        "peak_rss_mib": peak,
        "recall_filtered": round(float(np.mean(recall_filtered)) if recall_filtered else 1.0, 4),
        "recall_unfiltered": round(float(np.mean(recall_unfiltered)) if recall_unfiltered else 1.0, 4),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="MiniVectorDB benchmark")
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--mode", default="exact", choices=["exact", "ann"])
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args(argv)
    report = run(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
