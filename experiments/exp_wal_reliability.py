"""Does the log keep every commit, and does the reader keep the log?

Two properties of the write ahead log decide whether a commit that was
acknowledged can be rebuilt after a crash:

* every commit has to be in the log, and
* a reader that starts behind the records of the catalog has to leave the
  log alone.

The script writes five commits the way a checkpointed run does (write the
commit, then trim the log at the checkpoint), and then reads the log the way
a recovery does, starting at the position of the catalog.

It only uses the interface that both the old and the new log have, so the
same script runs on both.
"""

from __future__ import annotations

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

    for number in range(1, 6):
        writes = WriteSet()
        writes.add("t/docs/alive.bits", number - 1, bytes([number]))
        log.append_commit(number, writes, meta)
        # This is what a checkpoint does: drop the records the data files
        # already cover.
        log.trim(log.offset)
    print("commits acknowledged  : 5")
    print("log bytes on the disk  : %d" % path.stat().st_size)

    records = list(log.iter_records(0))
    print("records really in it   : %d" % len(records))
    log.close()

    # A recovery that starts at the position of the catalog: the records
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
