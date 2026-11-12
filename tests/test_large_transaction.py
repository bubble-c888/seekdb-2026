#!/usr/bin/env python3
"""A single transaction with many rows.

Two things used to be quadratic in the number of rows of a transaction: the
duplicate check of the primary key walked over every row that was inserted
before, and the patch of the alive bitmap walked over every pending change.
A case that puts a lot of rows into one transaction ran into the time limit
of a single request, so both are checked here.
"""

import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from minivectordb.engine.database import Database  # noqa: E402
from minivectordb.engine.executor import Executor  # noqa: E402
from minivectordb.errors import DuplicateKeyError  # noqa: E402
from minivectordb.sqlparse import parse  # noqa: E402

ROWS = 20000


class LargeTransactionTest(unittest.TestCase):
    """Many rows in one transaction stay inside the time limit."""

    def setUp(self) -> None:
        self.data_dir = Path(tempfile.mkdtemp(prefix="minivectordb-bigtxn-"))
        self.database = Database(self.data_dir)
        self.database.open()
        self.executor = Executor(self.database)

    def tearDown(self) -> None:
        try:
            self.database.close()
        finally:
            shutil.rmtree(self.data_dir, ignore_errors=True)

    def _execute(self, sql):
        return self.executor.execute(parse(sql))

    def test_many_rows_in_one_transaction(self) -> None:
        self._execute("CREATE TABLE t (id BIGINT PRIMARY KEY, g BIGINT);")
        self._execute("BEGIN;")
        began = time.perf_counter()
        for start in range(0, ROWS, 1000):
            values = ",".join(
                "({},{})".format(index, index % 3) for index in range(start, start + 1000)
            )
            self._execute("INSERT INTO t VALUES " + values)
        inserted = time.perf_counter() - began
        self._execute("COMMIT;")
        # The quadratic walks are gone; the limit of a request is 60 seconds.
        self.assertLess(inserted, 10.0, "the insert took {:.1f} s".format(inserted))
        rows = self._execute("SELECT id FROM t ORDER BY id;")["rows"]
        self.assertEqual(len(rows), ROWS)
        self.assertEqual(rows[0], [0])
        self.assertEqual(rows[-1], [ROWS - 1])

    def test_a_duplicate_inside_the_transaction_is_reported(self) -> None:
        self._execute("CREATE TABLE t (id BIGINT PRIMARY KEY, g BIGINT);")
        self._execute("BEGIN;")
        self._execute("INSERT INTO t VALUES (1, 1);")
        with self.assertRaises(DuplicateKeyError):
            self._execute("INSERT INTO t VALUES (2, 2), (1, 3);")
        # The failed statement left nothing behind, the first row is still there.
        rows = self._execute("SELECT id FROM t;")["rows"]
        self.assertEqual(rows, [[1]])
        self._execute("ROLLBACK;")
        self.assertEqual(self._execute("SELECT id FROM t;")["rows"], [])


if __name__ == "__main__":
    unittest.main()
