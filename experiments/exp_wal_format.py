"""The write ahead log in the binary format, measured against the old one.

The script writes the same batch of rows twice: once with the binary format
of the log and once with the text format it replaced (one JSON object per
line, the patches base64 encoded).  It reports how big the log of the batch
is in both formats, how often the file system is asked to flush, and what
the data directory costs.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, "/app")

from minivectordb.engine import storage as storage_module  # noqa: E402
from minivectordb.engine.database import Database  # noqa: E402
from minivectordb.engine.executor import Executor  # noqa: E402
from minivectordb.engine.storage import WriteSet  # noqa: E402
from minivectordb.sqlparse import parse  # noqa: E402

#: The magic of the binary format.  The build that still writes the text
#: format does not have it.
WAL_MAGIC = getattr(storage_module, "WAL_MAGIC", b"MVDBWAL1")

ROWS = 10000
DIM = 128
BATCH = 100


def fsync_counter():
    """Count the calls to fsync while the block runs."""
    original = os.fsync
    state = {"count": 0}

    def counting(fileno):
        state["count"] += 1
        return original(fileno)

    os.fsync = counting
    return state, original


def main() -> int:
    directory = Path("/data/walformat")
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True, exist_ok=True)

    import random

    random.seed(20261111)
    vectors = [[random.uniform(-1.0, 1.0) for _ in range(DIM)] for _ in range(ROWS)]

    database = Database(directory)
    database.open()
    executor = Executor(database)
    executor.execute(parse("CREATE TABLE docs (id BIGINT PRIMARY KEY, g BIGINT, v VECTOR(128));"))

    state, original = fsync_counter()
    began = time.perf_counter()
    for start in range(0, ROWS, BATCH):
        values = ",".join(
            "({},{},'[{}]')".format(
                index,
                index % 3,
                ",".join("%.6f" % value for value in vectors[index]),
            )
            for index in range(start, min(start + BATCH, ROWS))
        )
        executor.execute(parse("INSERT INTO docs VALUES " + values))
    write_seconds = time.perf_counter() - began
    os.fsync = original

    commits = ROWS // BATCH
    wal = sorted((directory / "wal").glob("wal-*.log"))
    wal_bytes = sum(path.stat().st_size for path in wal)
    data_bytes = sum(
        path.stat().st_size for path in directory.rglob("*") if path.is_file()
    ) - wal_bytes

    print("rows %d, batches %d, dim %d" % (ROWS, commits, DIM))
    print("write time            : %.3f s" % write_seconds)
    print("fsync calls           : %d (%.2f per batch)" % (state["count"], state["count"] / commits))
    print("log bytes on the disk : %d" % wal_bytes)
    print("data bytes            : %d" % data_bytes)
    for path in wal:
        head = path.read_bytes()[: len(WAL_MAGIC)]
        print("log %s : %d bytes, magic %r" % (path.name, path.stat().st_size, head))

    # The same record in the format it replaced: one JSON object per line,
    # the patches base64 encoded.
    writes = WriteSet()
    for index in range(BATCH):
        vector = np.asarray(vectors[index], dtype="float32")
        writes.add("t/docs/col_v.dat", index * DIM * 4, vector.tobytes())
    record = {
        "op": "commit",
        "txn": 1,
        "writes": writes.encode(),
        "meta": {"version": 1, "tables": {"docs": {"n_slots": BATCH}}},
    }
    text = json.dumps(record, ensure_ascii=False).encode("utf-8") + b"\n"
    print("one batch, text format : %d bytes" % len(text))

    store = storage_module.FileStore(directory / "compare")
    log = storage_module.WriteAheadLog(store, "wal")
    log.open()
    log.append_commit(1, writes, {"version": 1, "tables": {"docs": {"n_slots": BATCH}}})
    log.close()
    binary = (directory / "compare" / "wal" / "wal-000001.log").stat().st_size
    print("one batch, binary      : %d bytes (%.0f%% of the text format)"
          % (binary, 100.0 * binary / len(text)))

    database.close()
    shutil.rmtree(directory, ignore_errors=True)
    return 0


if __name__ == "__main__":
    import numpy as np  # noqa: E402

    raise SystemExit(main())
