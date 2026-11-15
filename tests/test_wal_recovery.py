"""Tests for the write ahead log and the recovery.

The log is the only thing that protects a commit between the moment it is
acknowledged and the moment the data files reach the disk.  The tests below
cover the cases in which that protection used to fail:

* the data files are lost after the log was written (the commit is replayed),
* the process dies while it recovers (the second recovery finishes the work),
* the log holds a record that a crash cut in half (the record is dropped),
* the log was written by an older version (the text format is still read).
"""

from __future__ import annotations

import json
import shutil
import struct
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from minivectordb.engine import storage as storage_module  # noqa: E402
from minivectordb.engine.database import Database  # noqa: E402
from minivectordb.engine.executor import Executor  # noqa: E402
from minivectordb.engine.storage import WAL_MAGIC, WriteAheadLog, WriteSet  # noqa: E402
from minivectordb.sqlparse import parse  # noqa: E402


class WalTestCase(unittest.TestCase):
    """Base class that gives every test its own empty data directory."""

    def setUp(self) -> None:
        self.data_dir = Path(tempfile.mkdtemp(prefix="minivectordb-wal-"))
        self.db = Database(self.data_dir)
        self.db.open()
        self.executor = Executor(self.db)

    def tearDown(self) -> None:
        try:
            if self.db is not None:
                self.db.close()
        except Exception:
            pass
        shutil.rmtree(self.data_dir, ignore_errors=True)

    def run_sql(self, sql: str):
        return self.executor.execute(parse(sql))

    def rows(self, sql: str):
        return self.run_sql(sql)["rows"]

    def crash(self) -> None:
        """Simulate a crash: the log is on the disk, the program is gone.

        The log is flushed, but the database is neither closed nor
        checkpointed, so the commits since the last checkpoint are only in the
        log.
        """
        self.db.wal.close()
        self.db = None

    def reopen(self) -> None:
        """Open the same directory again, like a restart of the program."""
        if self.db is not None:
            self.db.close()
        self.db = Database(self.data_dir)
        self.db.open()
        self.executor = Executor(self.db)

    def log_path(self) -> Path:
        return self.data_dir / "wal" / self.db.wal.segment

    def wal_files(self):
        directory = self.data_dir / "wal"
        if not directory.exists():
            return []
        return sorted(directory.glob("wal-*.log"))


class LogFormatTest(WalTestCase):
    def test_the_log_is_binary(self) -> None:
        self.run_sql("CREATE TABLE t (id BIGINT PRIMARY KEY, e VECTOR(4));")
        self.run_sql("INSERT INTO t VALUES (1, '[1, 2, 3, 4]');")
        path = self.log_path()
        self.crash()
        data = path.read_bytes()
        self.assertTrue(data.startswith(WAL_MAGIC))
        self.assertNotIn(b'"op"', data)

    def test_a_record_survives_a_round_trip(self) -> None:
        store = storage_module.FileStore(self.data_dir)
        log = WriteAheadLog(store, "wal")
        log.open()
        writes = WriteSet()
        writes.add("t/docs/col_id.dat", 0, b"\x01\x00\x00\x00\x00\x00\x00\x00")
        writes.add("t/docs/alive.bits", 3, b"\x05")
        meta = {"version": 1, "tables": {"docs": {"n_slots": 1}}}
        log.append_commit(7, writes, meta)
        log.close()

        reader = WriteAheadLog(store, "wal")
        reader.open()
        records = list(reader.iter_records(0))
        reader.close()
        self.assertEqual(len(records), 1)
        _offset, record = records[0]
        self.assertEqual(record["op"], "commit")
        self.assertEqual(record["txn"], 7)
        self.assertEqual(record["meta"], meta)
        self.assertEqual(
            record["writes"],
            [
                ("t/docs/col_id.dat", 0, b"\x01\x00\x00\x00\x00\x00\x00\x00"),
                ("t/docs/alive.bits", 3, b"\x05"),
            ],
        )

    def test_the_text_format_is_still_read(self) -> None:
        """A log of an older version is one JSON object per line."""
        store = storage_module.FileStore(self.data_dir)
        writes = WriteSet()
        writes.add("t/docs/alive.bits", 0, b"\x03")
        record = {
            "op": "commit",
            "txn": 1,
            "writes": writes.encode(),
            "meta": {"version": 1, "tables": {}},
        }
        payload = json.dumps(record, ensure_ascii=False).encode("utf-8") + b"\n"
        path = self.data_dir / "wal" / "wal-000001.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

        reader = WriteAheadLog(store, "wal")
        reader.open()
        records = list(reader.iter_records(0))
        reader.close()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0][1]["writes"], [("t/docs/alive.bits", 0, b"\x03")])

    def test_a_record_that_was_cut_in_half_is_dropped(self) -> None:
        self.run_sql("CREATE TABLE t (id BIGINT PRIMARY KEY, v BIGINT);")
        self.run_sql("INSERT INTO t VALUES (1, 10);")
        path = self.log_path()
        self.crash()
        size = path.stat().st_size
        with open(path, "ab") as handle:
            handle.write(b"\x40\x00\x00\x00partial record")

        self.reopen()
        self.assertEqual(self.rows("SELECT id, v FROM t;"), [[1, 10]])
        # The torn tail is cut away, so the next append starts at the end.
        self.assertEqual(path.stat().st_size, size)


class RecoveryTest(WalTestCase):
    def wipe_data_files(self) -> None:
        """The data files did not reach the disk, only the log did."""
        for path in (self.data_dir / "t" / "t").glob("*.dat"):
            path.write_bytes(b"")
        (self.data_dir / "t" / "t" / "alive.bits").write_bytes(b"")

    def test_a_commit_is_replayed_when_the_data_files_are_lost(self) -> None:
        """The log alone is enough to restore a commit."""
        self.run_sql("CREATE TABLE t (id BIGINT PRIMARY KEY, v BIGINT);")
        self.run_sql("INSERT INTO t VALUES (1, 10), (2, 20);")
        self.crash()
        self.wipe_data_files()

        self.reopen()
        self.assertEqual(self.rows("SELECT id, v FROM t;"), [[1, 10], [2, 20]])

    def test_the_recovery_is_idempotent(self) -> None:
        self.run_sql("CREATE TABLE t (id BIGINT PRIMARY KEY, v BIGINT);")
        self.run_sql("INSERT INTO t VALUES (1, 10);")
        self.crash()
        self.wipe_data_files()

        for _ in range(3):
            self.reopen()
            self.assertEqual(self.rows("SELECT id, v FROM t;"), [[1, 10]])
            self.db.wal.close()

    def test_a_recovery_that_is_killed_half_way_is_finished(self) -> None:
        """The recovery may be interrupted; the next one has to finish it."""
        self.run_sql("CREATE TABLE t (id BIGINT PRIMARY KEY, v BIGINT);")
        for value in range(1, 6):
            self.run_sql("INSERT INTO t VALUES (%d, %d);" % (value, value * 10))
        self.crash()
        self.wipe_data_files()

        original_apply = storage_module.FileStore.apply_writes
        state = {"failed": False}

        def exploding_apply(self, writes):
            if not state["failed"]:
                state["failed"] = True
                raise OSError("the process died during the recovery")
            return original_apply(self, writes)

        storage_module.FileStore.apply_writes = exploding_apply
        try:
            with self.assertRaises(OSError):
                Database(self.data_dir).open()
        finally:
            storage_module.FileStore.apply_writes = original_apply

        # The second recovery has to complete the work of the first one.
        self.reopen()
        self.assertEqual(
            self.rows("SELECT id, v FROM t;"),
            [[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]],
        )

    def test_an_uncommitted_transaction_is_dropped(self) -> None:
        self.run_sql("CREATE TABLE t (id BIGINT PRIMARY KEY, v BIGINT);")
        self.run_sql("INSERT INTO t VALUES (1, 10);")
        self.run_sql("BEGIN;")
        self.run_sql("INSERT INTO t VALUES (2, 20);")
        self.crash()
        self.wipe_data_files()

        self.reopen()
        self.assertEqual(self.rows("SELECT id, v FROM t;"), [[1, 10]])

    def test_the_log_does_not_grow_without_a_checkpoint(self) -> None:
        """A checkpoint drops the log that the data files already cover."""
        self.run_sql("CREATE TABLE t (id BIGINT PRIMARY KEY, v BIGINT);")
        for value in range(1, 41):
            self.run_sql("INSERT INTO t VALUES (%d, %d);" % (value, value))
        self.db.close()
        files = self.wal_files()
        self.assertEqual(len(files), 1)
        self.assertLess(files[0].stat().st_size, 8192)


class CheckpointTest(WalTestCase):
    def test_a_commit_survives_a_restart_without_a_checkpoint(self) -> None:
        """The commits since the last checkpoint are replayed."""
        self.run_sql("CREATE TABLE t (id BIGINT PRIMARY KEY, v BIGINT);")
        self.run_sql("INSERT INTO t VALUES (1, 10);")
        # The checkpoint is far away, so the commit is only in the log.
        self.db._pending_commits = 0
        self.run_sql("INSERT INTO t VALUES (2, 20);")
        self.assertTrue(self.db._pending_commits >= 1)

        self.reopen()
        self.assertEqual(self.rows("SELECT id, v FROM t;"), [[1, 10], [2, 20]])

    def test_a_vector_table_survives_a_restart(self) -> None:
        self.run_sql("CREATE TABLE docs (id BIGINT PRIMARY KEY, e VECTOR(3));")
        self.run_sql("INSERT INTO docs VALUES (1, '[1, 2, 3]'), (2, '[4, 5, 6]');")
        self.reopen()
        self.assertEqual(
            self.rows("SELECT id FROM docs ORDER BY L2_DISTANCE(e, '[0, 0, 0]') LIMIT 1;"),
            [[1]],
        )

    def test_a_varchar_table_survives_a_restart(self) -> None:
        self.run_sql("CREATE TABLE docs (id BIGINT PRIMARY KEY, s VARCHAR(16));")
        self.run_sql("INSERT INTO docs VALUES (1, '甲'), (2, '乙');")
        self.reopen()
        self.assertEqual(self.rows("SELECT id, s FROM docs ORDER BY id;"), [[1, "甲"], [2, "乙"]])


if __name__ == "__main__":
    unittest.main()
