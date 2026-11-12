#!/usr/bin/env python3
"""The dictionary of a VARCHAR column.

The values of a column live in a dictionary whose blobs stay on the disk: a
table may hold more text than the memory budget of the container allows.
The tests cover the lookups, the equality test that only needs the code of
the literal, the order of the values and the values after a restart.
"""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from minivectordb.engine.database import Database  # noqa: E402
from minivectordb.engine.executor import Executor  # noqa: E402
from minivectordb.sqlparse import parse  # noqa: E402


class DictionaryTest(unittest.TestCase):
    """Lookups, ordering and the values on disk."""

    def setUp(self) -> None:
        self.data_dir = Path(tempfile.mkdtemp(prefix="minivectordb-dict-"))
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

    def test_the_values_are_read_back_from_the_files(self) -> None:
        self._execute("CREATE TABLE t (id BIGINT PRIMARY KEY, s VARCHAR(64));")
        values = ",".join(
            "({},'value-{}')".format(index, index) for index in range(50)
        )
        self._execute("INSERT INTO t VALUES " + values + ";")
        rows = self._execute("SELECT id, s FROM t ORDER BY id;")["rows"]
        self.assertEqual(len(rows), 50)
        self.assertEqual(rows[0], [0, "value-0"])
        self.assertEqual(rows[49], [49, "value-49"])

    def test_an_equality_only_needs_the_code(self) -> None:
        self._execute("CREATE TABLE t (id BIGINT PRIMARY KEY, s VARCHAR(64));")
        self._execute("INSERT INTO t VALUES (1,'alpha'),(2,'beta'),(3,'gamma');")
        rows = self._execute("SELECT id FROM t WHERE s = 'beta';")["rows"]
        self.assertEqual(rows, [[2]])
        # A value that is not in the dictionary matches nothing.
        rows = self._execute("SELECT id FROM t WHERE s = 'delta';")["rows"]
        self.assertEqual(rows, [])
        # The comparison works inside a transaction as well.
        self._execute("BEGIN;")
        self._execute("INSERT INTO t VALUES (4,'delta');")
        rows = self._execute("SELECT id FROM t WHERE s = 'delta';")["rows"]
        self.assertEqual(rows, [[4]])
        self._execute("ROLLBACK;")

    def test_the_values_are_ordered_by_the_code_point_order(self) -> None:
        self._execute("CREATE TABLE t (id BIGINT PRIMARY KEY, s VARCHAR(64));")
        self._execute(
            "INSERT INTO t VALUES (1,'乙'),(2,'甲'),(3,'a'),(4,'A'),(5,'中');"
        )
        rows = self._execute("SELECT s FROM t ORDER BY s;")["rows"]
        self.assertEqual(
            [row[0] for row in rows], ["A", "a", "中", "乙", "甲"]
        )
        # A WHERE clause without an ORDER BY returns the rows in the order of
        # the table, only the values that are greater are kept.
        rows = self._execute("SELECT id FROM t WHERE s > 'a';")["rows"]
        self.assertEqual([row[0] for row in rows], [1, 2, 5])

    def test_the_values_survive_a_restart(self) -> None:
        self._execute("CREATE TABLE t (id BIGINT PRIMARY KEY, s VARCHAR(64));")
        self._execute("INSERT INTO t VALUES (1,'first'),(2,'second');")
        self.database.close()

        database = Database(self.data_dir)
        database.open()
        try:
            executor = Executor(database)
            rows = executor.execute(parse("SELECT id, s FROM t ORDER BY id;"))["rows"]
            self.assertEqual(rows, [[1, "first"], [2, "second"]])
            rows = executor.execute(parse("SELECT id FROM t WHERE s = 'second';"))["rows"]
            self.assertEqual(rows, [[2]])
        finally:
            database.close()

    def test_the_dictionary_does_not_hold_every_value(self) -> None:
        self._execute("CREATE TABLE t (id BIGINT PRIMARY KEY, s VARCHAR(80));")
        values = ",".join(
            "({},'{}')".format(index, "x" * 64 + str(index)) for index in range(200)
        )
        self._execute("INSERT INTO t VALUES " + values + ";")
        table = self.database.table("t")
        dictionary = table.dictionary(table.columns[1])
        self.assertEqual(len(dictionary), 200)
        # The values are read back from the files, the cache stays small.
        self.assertLessEqual(dictionary._cache_bytes, dictionary.CACHE_BYTES)
        self.assertEqual(dictionary.text_of(7), "x" * 64 + "7")

    def test_a_long_value_is_stored_and_read_back(self) -> None:
        self._execute("CREATE TABLE t (id BIGINT PRIMARY KEY, s VARCHAR(4096));")
        long_text = "y" * 4096
        self._execute(
            "INSERT INTO t VALUES (1,'{}');".format(long_text)
        )
        rows = self._execute("SELECT s FROM t;")["rows"]
        self.assertEqual(rows[0][0], long_text)
        rows = self._execute("SELECT id FROM t WHERE s = '{}';".format(long_text))["rows"]
        self.assertEqual(rows, [[1]])


if __name__ == "__main__":
    unittest.main()
