"""The log of the commit: how many records and how many bytes.

The script works with both versions of the log: the older one held the
patches of a commit, the newer one holds the state of a commit.  It writes
five commits the way a checkpointed run does (write the commit, then cut
the log at the checkpoint), counts the records that are really in the log,
and lets a reader start at the position of the catalog to see whether the
log survives that.
"""

import inspect
import shutil
import sys
from pathlib import Path

sys.path.insert(0, "/app")

from minivectordb.engine.storage import FileStore, WriteAheadLog, WriteSet  # noqa: E402


def main() -> int:
    directory = Path("/data/walreliability")
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True, exist_ok=True)

    store = FileStore(directory)
    log = WriteAheadLog(store, "wal")
    log.open()
    path = store.path(log.key)
    meta = {"version": 1, "tables": {}}
    holds_state = "marker" in inspect.signature(log.append_commit).parameters

    for number in range(1, 6):
        if holds_state:
            marker = {"sizes": {"t/docs/alive.bits": number}}
            log.append_commit(number, marker, meta)
        else:
            writes = WriteSet()
            writes.add("t/docs/alive.bits", 0, bytes([number]))
            log.append_commit(number, writes, meta)
        # This is what a checkpoint does: cut the log at the checkpoint.
        log.trim(log.offset)

    print("commits acknowledged  : 5")
    print("log bytes on the disk  : %d" % path.stat().st_size)
    records = list(log.iter_records(0))
    print("records really in it   : %d" % len(records))
    log.close()

    # A reader that starts at the position of the catalog: the records
    # before it are already in the data files.
    reader = WriteAheadLog(store, "wal")
    reader.open()
    before = path.stat().st_size
    list(reader.iter_records(before))
    after = path.stat().st_size
    reader.close()
    print("log bytes before/after that recovery: %d / %d" % (before, after))

    shutil.rmtree(directory, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
