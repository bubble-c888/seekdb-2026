"""Tests for the write ahead log and the recovery.

The log holds the state of a commit and not the data of a commit: the data
is in the data files, and the log says how long every file was after the
commit.  A recovery therefore reads the log, takes the last metadata and
cuts the files back to the lengths the log names, so that the part of a
batch that was never committed is gone.

The tests cover that behaviour, the cases in which a crash used to lose a
row that had been acknowledged, and the format of the log itself.
"""

from __future__ import annotations

import json
import shutil
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
from minivectordb.engine.storage import WAL_MAGIC, WriteAheadLog  # noqa: E402
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
        """Simulate a crash: the log is flushed, but the database is gone.

        The log is flushed, but the database is neither closed nor
        checkpointed, so the commits since the last checkpoint are only in
        the log.
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
        marker = {"sizes": {"t/docs/col_id.dat": 4096, "t/docs/alive.bits": 13}}
        meta = {"version": 1, "tables": {"docs": {"n_slots": 1}}}
        log.append_commit(7, marker, meta)
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
        self.assertEqual(record["marker"], marker)

    def test_the_log_holds_the_state_and_not_the_data(self) -> None:
        """The log must not hold a second copy of the rows."""
        self.run_sql("CREATE TABLE t (id BIGINT PRIMARY KEY, e VECTOR(128));")
        for number in range(1, 101):
            self.run_sql(
                "INSERT INTO t VALUES (%d, '[%s]');"
                % (number, ",".join(["0.5"] * 128))
            )
        path = self.log_path()
        self.crash()
        data = path.read_bytes()
        # A hundred rows of 128 floats are 51200 bytes of data; the log holds
        # the state of every commit and not the rows themselves.
        self.assertLess(len(data), 20000)

    def test_the_text_format_is_still_read(self) -> None:
        """A log of an older version is one JSON object per line."""
        store = storage_module.FileStore(self.data_dir)
        record = {
            "op": "commit",
            "txn": 1,
            "writes": [["t/docs/alive.bits", 0, "AQ=="]],
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
        self.assertEqual(records[0][1]["writes"], [("t/docs/alive.bits", 0, b"\x01")])

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
    def test_the_committed_rows_are_there_after_a_restart(self) -> None:
        self.run_sql("CREATE TABLE t (id BIGINT PRIMARY KEY, v BIGINT);")
        self.run_sql("INSERT INTO t VALUES (1, 10), (2, 20);")
        self.crash()
        self.reopen()
        self.assertEqual(self.rows("SELECT id, v FROM t;"), [[1, 10], [2, 20]])

    def test_a_batch_that_was_not_logged_is_cut_away(self) -> None:
        """The data of a commit that never reached the log is not visible."""
        self.run_sql("CREATE TABLE t (id BIGINT PRIMARY KEY, v BIGINT);")
        self.run_sql("INSERT INTO t VALUES (1, 10);")

        # The commit writes the data into the files and then fails to reach
        # the log: this is the window between the two.
        original = storage_module.WriteAheadLog.append_commit

        def failing_append(self, txn_id, marker, meta):
            raise OSError("the process died before the log was written")

        storage_module.WriteAheadLog.append_commit = failing_append
        try:
            with self.assertRaises(OSError):
                self.run_sql("INSERT INTO t VALUES (2, 20);")
        finally:
            storage_module.WriteAheadLog.append_commit = original

        # The row is in the file, but it was never committed.  The process
        # died, so nothing was closed and the catalog was not written.
        self.crash()
        self.reopen()
        self.assertEqual(self.rows("SELECT id, v FROM t;"), [[1, 10]])

    def test_an_uncommitted_transaction_is_dropped(self) -> None:
        self.run_sql("CREATE TABLE t (id BIGINT PRIMARY KEY, v BIGINT);")
        self.run_sql("INSERT INTO t VALUES (1, 10);")
        self.run_sql("BEGIN;")
        self.run_sql("INSERT INTO t VALUES (2, 20);")
        self.crash()
        self.reopen()
        self.assertEqual(self.rows("SELECT id, v FROM t;"), [[1, 10]])

    def test_a_deleted_row_that_was_not_committed_comes_back(self) -> None:
        """The bitmap of a row that was deleted but not committed is put back."""
        self.run_sql("CREATE TABLE t (id BIGINT PRIMARY KEY, v BIGINT);")
        self.run_sql("INSERT INTO t VALUES (1, 10), (2, 20);")

        # The delete reaches the file but not the log.
        original = storage_module.WriteAheadLog.append_commit

        def failing_append(self, txn_id, marker, meta):
            raise OSError("the process died before the log was written")

        storage_module.WriteAheadLog.append_commit = failing_append
        try:
            with self.assertRaises(OSError):
                self.run_sql("DELETE FROM t WHERE id = 1;")
        finally:
            storage_module.WriteAheadLog.append_commit = original

        self.crash()
        self.reopen()
        self.assertEqual(self.rows("SELECT id, v FROM t;"), [[1, 10], [2, 20]])

    def test_a_deleted_row_that_was_committed_stays_deleted(self) -> None:
        self.run_sql("CREATE TABLE t (id BIGINT PRIMARY KEY, v BIGINT);")
        self.run_sql("INSERT INTO t VALUES (1, 10), (2, 20);")
        self.run_sql("DELETE FROM t WHERE id = 1;")
        self.crash()
        self.reopen()
        self.assertEqual(self.rows("SELECT id, v FROM t;"), [[2, 20]])

    def test_the_recovery_is_idempotent(self) -> None:
        self.run_sql("CREATE TABLE t (id BIGINT PRIMARY KEY, v BIGINT);")
        self.run_sql("INSERT INTO t VALUES (1, 10);")
        self.crash()
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

        original = storage_module.FileStore.write

        def exploding_write(self, key, offset, data):
            raise OSError("the process died during the recovery")

        storage_module.FileStore.write = exploding_write
        try:
            with self.assertRaises(OSError):
                Database(self.data_dir).open()
        finally:
            storage_module.FileStore.write = original

        # The second recovery has to complete the work of the first one.
        self.reopen()
        self.assertEqual(
            self.rows("SELECT id, v FROM t;"),
            [[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]],
        )

    def test_the_log_does_not_grow_without_a_checkpoint(self) -> None:
        """A checkpoint drops the log that the data files already cover."""
        self.run_sql("CREATE TABLE t (id BIGINT PRIMARY KEY, v BIGINT);")
        for value in range(1, 41):
            self.run_sql("INSERT INTO t VALUES (%d, %d);" % (value, value))
        self.db.close()
        files = self.wal_files()
        # The segment the catalog names and the one that follows it.
        self.assertEqual(len(files), 2)
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
