#!/usr/bin/env python3
"""A transaction has to see its own changes.

The task statement asks for it twice: "事务内读到自己的修改" and
"表数据与向量检索一致".  The public cases only read after a commit or after
a rollback, so the behaviour inside the transaction needs a test of its own.
"""

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


class TransactionVisibilityTest(unittest.TestCase):
    """The rows of the running transaction are part of the result."""

    def setUp(self) -> None:
        self.data_dir = Path(tempfile.mkdtemp(prefix="minivectordb-txn-"))
        self.database = Database(self.data_dir)
        self.database.open()
        self.executor = Executor(self.database)
        self._execute(
            "CREATE TABLE docs (id BIGINT PRIMARY KEY, category VARCHAR(8), "
            "content VARCHAR(16), embedding VECTOR(2));"
        )
        self._execute(
            "INSERT INTO docs VALUES (1,'a','first','[0, 0]'),"
            "(2,'b','second','[5, 0]'),(3,'a','third','[9, 0]');"
        )

    def tearDown(self) -> None:
        try:
            self.database.close()
        finally:
            shutil.rmtree(self.data_dir, ignore_errors=True)

    def _execute(self, sql):
        return self.executor.execute(parse(sql))

    def _rows(self, sql):
        return self._execute(sql)["rows"]

    # -- the plain columns ----------------------------------------------
    def test_an_inserted_row_is_visible(self) -> None:
        self._execute("BEGIN;")
        self._execute("INSERT INTO docs VALUES (4,'c','fourth','[1, 0]');")
        self.assertEqual(
            self._rows("SELECT id FROM docs ORDER BY id;"), [[1], [2], [3], [4]]
        )
        self.assertEqual(
            self._rows("SELECT id FROM docs WHERE category = 'c';"), [[4]]
        )
        self._execute("ROLLBACK;")
        self.assertEqual(self._rows("SELECT id FROM docs ORDER BY id;"), [[1], [2], [3]])

    def test_a_deleted_row_is_gone(self) -> None:
        self._execute("BEGIN;")
        self._execute("DELETE FROM docs WHERE id = 2;")
        self.assertEqual(self._rows("SELECT id FROM docs ORDER BY id;"), [[1], [3]])
        self.assertEqual(self._rows("SELECT id FROM docs WHERE id = 2;"), [])
        self._execute("ROLLBACK;")
        self.assertEqual(self._rows("SELECT id FROM docs ORDER BY id;"), [[1], [2], [3]])

    def test_an_updated_row_shows_the_new_value(self) -> None:
        self._execute("BEGIN;")
        self._execute("UPDATE docs SET content = 'changed' WHERE id = 1;")
        self.assertEqual(
            self._rows("SELECT id, content FROM docs ORDER BY id;"),
            [[1, "changed"], [2, "second"], [3, "third"]],
        )
        self.assertEqual(
            self._rows("SELECT id FROM docs WHERE content = 'changed';"), [[1]]
        )
        self.assertEqual(
            self._rows("SELECT id FROM docs WHERE content = 'first';"), []
        )
        self._execute("ROLLBACK;")
        self.assertEqual(
            self._rows("SELECT id, content FROM docs ORDER BY id;"),
            [[1, "first"], [2, "second"], [3, "third"]],
        )

    # -- the vector retrieval --------------------------------------------
    def test_the_vector_search_sees_the_new_row(self) -> None:
        self._execute("BEGIN;")
        self._execute("INSERT INTO docs VALUES (4,'c','fourth','[0, 1]');")
        rows = self._rows(
            "SELECT id, L2_DISTANCE(embedding, '[0, 0]') AS distance "
            "FROM docs ORDER BY distance LIMIT 2;"
        )
        self.assertEqual([row[0] for row in rows], [1, 4])
        self.assertAlmostEqual(rows[0][1], 0.0)
        self.assertAlmostEqual(rows[1][1], 1.0)
        self._execute("ROLLBACK;")

    def test_the_vector_search_sees_the_new_vector(self) -> None:
        self._execute("BEGIN;")
        self._execute("UPDATE docs SET embedding = '[0, 0]' WHERE id = 3;")
        rows = self._rows(
            "SELECT id, L2_DISTANCE(embedding, '[0, 0]') AS distance "
            "FROM docs ORDER BY distance LIMIT 3;"
        )
        # The row 3 is at the origin now, the row 1 was there before.
        self.assertEqual([row[0] for row in rows], [1, 3, 2])
        self.assertAlmostEqual(rows[1][1], 0.0)
        self.assertAlmostEqual(rows[2][1], 5.0)
        self._execute("ROLLBACK;")
        rows = self._rows(
            "SELECT id, L2_DISTANCE(embedding, '[0, 0]') AS distance "
            "FROM docs ORDER BY distance LIMIT 3;"
        )
        self.assertEqual([row[0] for row in rows], [1, 2, 3])
        self.assertAlmostEqual(rows[2][1], 9.0)

    def test_the_vector_search_does_not_see_a_deleted_row(self) -> None:
        self._execute("BEGIN;")
        self._execute("DELETE FROM docs WHERE id = 1;")
        rows = self._rows(
            "SELECT id, L2_DISTANCE(embedding, '[0, 0]') AS distance "
            "FROM docs ORDER BY distance LIMIT 3;"
        )
        self.assertEqual([row[0] for row in rows], [2, 3])
        self._execute("ROLLBACK;")

    def test_the_filter_sees_the_new_value(self) -> None:
        self._execute("BEGIN;")
        self._execute("UPDATE docs SET category = 'z' WHERE id = 1;")
        self.assertEqual(
            self._rows("SELECT id FROM docs WHERE category = 'z';"), [[1]]
        )
        self.assertEqual(
            self._rows("SELECT id FROM docs WHERE category = 'a';"), [[3]]
        )
        self._execute("ROLLBACK;")

    # -- the commit ------------------------------------------------------
    def test_the_changes_survive_the_commit(self) -> None:
        self._execute("BEGIN;")
        self._execute("INSERT INTO docs VALUES (4,'c','fourth','[1, 0]');")
        self._execute("UPDATE docs SET content = 'changed' WHERE id = 1;")
        self._execute("DELETE FROM docs WHERE id = 2;")
        self._execute("COMMIT;")
        self.assertEqual(
            self._rows("SELECT id, content FROM docs ORDER BY id;"),
            [[1, "changed"], [3, "third"], [4, "fourth"]],
        )
        rows = self._rows(
            "SELECT id, L2_DISTANCE(embedding, '[0, 0]') AS distance "
            "FROM docs ORDER BY distance LIMIT 2;"
        )
        self.assertEqual([row[0] for row in rows], [1, 4])


class AnnTransactionTest(unittest.TestCase):
    """The same in the approximate mode.

    The approximate path is used for the large data sets, so the limit is
    lowered here to reach it with a few rows.
    """

    def setUp(self) -> None:
        from minivectordb.engine import executor as executor_module

        self.module = executor_module
        self.original = executor_module.ANN_EXACT_LIMIT
        executor_module.ANN_EXACT_LIMIT = 0
        self.data_dir = Path(tempfile.mkdtemp(prefix="minivectordb-ann-txn-"))
        self.database = Database(self.data_dir)
        self.database.open()
        self.executor = Executor(self.database)
        self._execute(
            "CREATE TABLE docs (id BIGINT PRIMARY KEY, category VARCHAR(8), "
            "content VARCHAR(16), embedding VECTOR(2));"
        )
        self._execute(
            "INSERT INTO docs VALUES (1,'a','first','[0, 0]'),"
            "(2,'b','second','[5, 0]'),(3,'a','third','[9, 0]');"
        )
        self._execute("CREATE VECTOR INDEX docs_vec ON docs(embedding);")
        self.database.mode = "ann"

    def tearDown(self) -> None:
        self.module.ANN_EXACT_LIMIT = self.original
        try:
            self.database.close()
        finally:
            shutil.rmtree(self.data_dir, ignore_errors=True)

    def _execute(self, sql):
        return self.executor.execute(parse(sql))

    def _rows(self, sql):
        return self._execute(sql)["rows"]

    def test_the_approximate_search_sees_the_transaction(self) -> None:
        self._execute("BEGIN;")
        self._execute("INSERT INTO docs VALUES (4,'c','fourth','[0, 1]');")
        self._execute("UPDATE docs SET embedding = '[0, 0]' WHERE id = 3;")
        self._execute("DELETE FROM docs WHERE id = 2;")
        rows = self._rows(
            "SELECT id, L2_DISTANCE(embedding, '[0, 0]') AS distance "
            "FROM docs ORDER BY distance LIMIT 5;"
        )
        found = [row[0] for row in rows]
        self.assertIn(4, found)
        self.assertNotIn(2, found)
        self._execute("ROLLBACK;")


if __name__ == "__main__":
    unittest.main()
