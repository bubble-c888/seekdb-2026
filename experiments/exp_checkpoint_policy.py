#!/usr/bin/env python3
"""What does the checkpoint on every commit cost, and what does it buy?

The submitted version writes the catalog and trims the log after every commit.
The comment in the engine says the reason is the recovery: without a fresh
checkpoint the log of a large table would have to be replayed at the start, and
the start has a time limit of 120 seconds.

The script measures both sides of that trade with the same workload:

* write time with the checkpoint on every commit (the submitted behaviour),
* write time without it (the log is kept, nothing is rewritten),
* recovery time after a crash in both cases.

The crash is a child process that exits without closing the database, so the
last state exists only in the write ahead log.

Usage::

    python exp_checkpoint_policy.py [--rows 10000] [--batch 100]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = str(Path(__file__).resolve().parent.parent)
WORK = Path(r"E:\MiniVectorDB\runs\checkpoint_exp")

LOAD = r'''
import json, os, random, sys, time
sys.path.insert(0, r"__REPO__")
from minivectordb.engine.database import Database
from minivectordb.engine import storage

mode = sys.argv[1]
directory = sys.argv[2]
rows = int(sys.argv[3])
batch = int(sys.argv[4])
dim = int(sys.argv[5])

if mode == "no-checkpoint":
    # Emulate a policy that checkpoints only every N commits: neither the
    # catalog nor the log is touched on the way, so the log grows.
    Database.checkpoint = lambda self: None
    storage.WriteAheadLog.trim = lambda self, offset: None

random.seed(20261111)
db = Database(directory)
db.open()
from minivectordb.engine.executor import Executor
from minivectordb.sqlparse import parse
executor = Executor(db)
db.execute = lambda sql: executor.execute(parse(sql))
db.execute("CREATE TABLE docs (id BIGINT PRIMARY KEY, g BIGINT, v VECTOR(%d))" % dim)
began = time.perf_counter()
for start in range(0, rows, batch):
    values = ",".join(
        "({0},{1},'[{2}]')".format(index, index % 3, ",".join("%.6f" % random.uniform(-1, 1) for _ in range(dim)))
        for index in range(start, min(start + batch, rows))
    )
    db.execute("INSERT INTO docs VALUES " + values)
write = time.perf_counter() - began
print(json.dumps({"mode": mode, "write_seconds": round(write, 3)}))
sys.stdout.flush()
# The crash: the process ends without closing the database, so nothing but the
# write ahead log holds the last state.
os._exit(0)
'''

RECOVER = r'''
import json, os, sys, time
sys.path.insert(0, r"__REPO__")
from minivectordb.engine.database import Database
directory = sys.argv[1]
began = time.perf_counter()
db = Database(directory)
db.open()
elapsed = time.perf_counter() - began
from minivectordb.engine.executor import Executor
from minivectordb.sqlparse import parse
executor = Executor(db)
count = executor.execute(parse("SELECT id FROM docs"))
print(json.dumps({"recover_seconds": round(elapsed, 3), "rows": len(count.get("rows", []))}))
sys.stdout.flush()
os._exit(0)
'''


def run(script, *arguments):
    result = subprocess.run(
        [sys.executable, "-c", script, *[str(value) for value in arguments]],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0 or not result.stdout.strip():
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise SystemExit("child failed: " + str(result.returncode))
    line = result.stdout.strip().splitlines()[-1]
    return json.loads(line)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("--dim", type=int, default=128)
    arguments = parser.parse_args()

    load = LOAD.replace("__REPO__", REPO)
    recover = RECOVER.replace("__REPO__", REPO)

    report = {}
    for mode in ("checkpoint", "no-checkpoint"):
        directory = WORK / mode
        if directory.exists():
            shutil.rmtree(directory, ignore_errors=True)
        directory.mkdir(parents=True, exist_ok=True)
        loaded = run(load, mode, directory, arguments.rows, arguments.batch, arguments.dim)
        recovered = run(recover, directory)
        size = sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
        report[mode] = {
            "write_seconds": loaded["write_seconds"],
            "recover_seconds": recovered["recover_seconds"],
            "rows_after_recovery": recovered["rows"],
            "data_dir_mib": round(size / 1024 / 1024, 2),
        }
        print(
            "{:14s} write {:6.3f}s   recover {:6.3f}s   rows {}   data {:6.2f} MiB".format(
                mode,
                loaded["write_seconds"],
                recovered["recover_seconds"],
                recovered["rows"],
                report[mode]["data_dir_mib"],
            )
        )

    print("\nSUMMARY " + json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
