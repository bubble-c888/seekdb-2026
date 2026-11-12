#!/usr/bin/env python3
"""The SQL forms that are next to the ones the public cases use.

The public cases are a sample; the formal ones use the same subset but they
may write it in another way.  Every statement here is inside the subset of
the task statement, so it has to work.
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
from minivectordb.errors import MiniVectorDBError  # noqa: E402
from minivectordb.sqlparse import parse  # noqa: E402


class SqlFormsTest(unittest.TestCase):
    """Statements of the subset that the public cases do not use."""

    def setUp(self) -> None:
        self.data_dir = Path(tempfile.mkdtemp(prefix="minivectordb-forms-"))
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

    def _rows(self, sql):
        return self._execute(sql)["rows"]

    def _setup(self):
        self._execute(
            "CREATE TABLE docs (id BIGINT PRIMARY KEY, category VARCHAR(8), "
            "content VARCHAR(16), embedding VECTOR(2));"
        )
        self._execute(
            "INSERT INTO docs VALUES (1,'a','first','[0, 0]'),"
            "(2,'b','second','[1, 0]'),(3,'a','third','[2, 0]'),"
            "(4,'b','fourth','[3, 0]'),(5,'c','fifth','[4, 0]');"
        )

    # -- LIMIT ----------------------------------------------------------
    def test_limit_with_an_offset(self) -> None:
        self._setup()
        self.assertEqual(self._rows("SELECT id FROM docs ORDER BY id LIMIT 2;"), [[1], [2]])
        self.assertEqual(
            self._rows("SELECT id FROM docs ORDER BY id LIMIT 2, 2;"), [[3], [4]]
        )
        # ``LIMIT count OFFSET offset``: the third row.
        self.assertEqual(
            self._rows("SELECT id FROM docs ORDER BY id LIMIT 1 OFFSET 2;"), [[3]]
        )
        # An offset behind the end of the table returns nothing.
        self.assertEqual(self._rows("SELECT id FROM docs ORDER BY id LIMIT 9, 2;"), [])

    def test_limit_with_an_offset_and_a_filter(self) -> None:
        self._setup()
        self.assertEqual(
            self._rows(
                "SELECT id FROM docs WHERE category = 'a' ORDER BY id LIMIT 1, 1;"
            ),
            [[3]],
        )

    # -- projections ----------------------------------------------------
    def test_select_star(self) -> None:
        self._setup()
        result = self._execute("SELECT * FROM docs WHERE id = 1;")
        self.assertEqual(result["columns"], ["id", "category", "content", "embedding"])
        self.assertEqual(
            result["rows"], [[1, "a", "first", [0.0, 0.0]]]
        )

    def test_an_alias_of_a_column(self) -> None:
        self._setup()
        result = self._execute("SELECT id AS key, content AS text FROM docs WHERE id = 1;")
        self.assertEqual(result["columns"], ["key", "text"])
        self.assertEqual(result["rows"], [[1, "first"]])

    def test_the_vector_is_projected_as_an_array(self) -> None:
        self._setup()
        result = self._execute("SELECT embedding FROM docs WHERE id = 2;")
        self.assertEqual(result["columns"], ["embedding"])
        self.assertEqual(result["rows"], [[[1.0, 0.0]]])

    # -- statements -----------------------------------------------------
    def test_insert_with_a_column_list(self) -> None:
        self._execute("CREATE TABLE docs (id BIGINT PRIMARY KEY, content VARCHAR(16));")
        self._execute("INSERT INTO docs (id, content) VALUES (7, 'seven');")
        self.assertEqual(self._rows("SELECT id, content FROM docs;"), [[7, "seven"]])

    def test_a_table_level_primary_key(self) -> None:
        self._execute("CREATE TABLE docs (id BIGINT, content VARCHAR(16), PRIMARY KEY (id));")
        self._execute("INSERT INTO docs VALUES (1, 'one');")
        self.assertEqual(self._rows("SELECT id FROM docs;"), [[1]])

    def test_an_empty_string(self) -> None:
        self._execute("CREATE TABLE docs (id BIGINT PRIMARY KEY, content VARCHAR(16));")
        self._execute("INSERT INTO docs VALUES (1, '');")
        self.assertEqual(self._rows("SELECT content FROM docs;"), [[""]])

    def test_a_table_without_a_vector_column(self) -> None:
        self._execute("CREATE TABLE docs (id BIGINT PRIMARY KEY, content VARCHAR(16));")
        self._execute("INSERT INTO docs VALUES (1, 'one');")
        self.assertEqual(self._rows("SELECT id FROM docs;"), [[1]])

    # -- conditions -----------------------------------------------------
    def test_update_and_delete_by_a_condition(self) -> None:
        self._setup()
        result = self._execute("UPDATE docs SET content = 'changed' WHERE category = 'a';")
        self.assertEqual(result["affected_rows"], 2)
        result = self._execute("DELETE FROM docs WHERE category = 'b';")
        self.assertEqual(result["affected_rows"], 2)
        self.assertEqual(self._rows("SELECT id FROM docs ORDER BY id;"), [[1], [3], [5]])

    def test_a_condition_on_a_string_column(self) -> None:
        self._setup()
        self.assertEqual(
            self._rows("SELECT id FROM docs WHERE content = 'third';"), [[3]]
        )
        # The strings are ordered by their code points, so 'third' is the only
        # value that is greater than 'second'.
        self.assertEqual(
            self._rows("SELECT id FROM docs WHERE content > 'second' ORDER BY id;"),
            [[3]],
        )
        self.assertEqual(
            self._rows("SELECT id FROM docs WHERE content <= 'second' ORDER BY id;"),
            [[1], [2], [4], [5]],
        )

    def test_a_condition_on_a_vector_column_is_rejected(self) -> None:
        self._setup()
        with self.assertRaises(MiniVectorDBError):
            self._execute("SELECT id FROM docs WHERE embedding = '[0, 0]';")

    # -- errors ---------------------------------------------------------
    def test_the_statements_outside_the_subset_are_rejected(self) -> None:
        self._setup()
        for statement in (
            "SELECT id FROM docs WHERE content IS NULL;",
            "SELECT a.id FROM docs a JOIN docs b ON a.id = b.id;",
            "SELECT COUNT(*) FROM docs;",
            "SELECT id FROM docs GROUP BY id;",
            "SELECT id FROM docs WHERE id IN (SELECT id FROM docs);",
            "ALTER TABLE docs ADD extra BIGINT;",
            "CREATE TABLE other (id BIGINT PRIMARY KEY, p BIGINT REFERENCES docs(id));",
            "UPDATE docs SET id = 9 WHERE id = 1;",
            "SELECT id FROM docs ORDER BY id;",
        ):
            with self.subTest(statement=statement):
                if statement == "SELECT id FROM docs ORDER BY id;":
                    # A statement of the subset, it has to work.
                    self.assertEqual(len(self._rows(statement)), 5)
                    continue
                with self.assertRaises(MiniVectorDBError):
                    self._execute(statement)


if __name__ == "__main__":
    unittest.main()
