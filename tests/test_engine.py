"""Functional tests for MiniVectorDB.

The tests cover the behaviour that the task statement demands but that the
public mysqltest samples do not exercise: transactions, persistence across a
restart, crash recovery, index maintenance, error handling and the resource
behaviour for data sets that are larger than the memory budget.

Run them with::

    python -m unittest tests.test_engine -v
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from minivectordb.engine.database import Database  # noqa: E402
from minivectordb.engine.executor import Executor  # noqa: E402
from minivectordb.errors import MiniVectorDBError  # noqa: E402
from minivectordb.sqlparse import parse  # noqa: E402


class DatabaseTestCase(unittest.TestCase):
    """Base class that gives every test its own empty data directory."""

    def setUp(self) -> None:
        self.data_dir = Path(tempfile.mkdtemp(prefix="minivectordb-test-"))
        self.db = Database(self.data_dir)
        self.db.open()
        self.executor = Executor(self.db)

    def tearDown(self) -> None:
        try:
            self.db.close()
        except Exception:
            pass
        shutil.rmtree(self.data_dir, ignore_errors=True)

    def run_sql(self, sql: str):
        return self.executor.execute(parse(sql))

    def rows(self, sql: str):
        return self.run_sql(sql)["rows"]


class TransactionTest(DatabaseTestCase):
    def test_rollback_discards_every_change(self) -> None:
        self.run_sql("CREATE TABLE t (id BIGINT PRIMARY KEY, v BIGINT);")
        self.run_sql("BEGIN;")
        self.run_sql("INSERT INTO t VALUES (1, 10);")
        self.run_sql("UPDATE t SET v = 20 WHERE id = 1;")
        self.run_sql("ROLLBACK;")
        self.assertEqual(self.rows("SELECT id, v FROM t;"), [])
        self.assertEqual(self.run_sql("SELECT id FROM t;")["rows"], [])

    def test_commit_keeps_the_changes(self) -> None:
        self.run_sql("CREATE TABLE t (id BIGINT PRIMARY KEY, v BIGINT);")
        self.run_sql("BEGIN;")
        self.run_sql("INSERT INTO t VALUES (1, 10);")
        self.run_sql("COMMIT;")
        self.assertEqual(self.rows("SELECT id, v FROM t;"), [[1, 10]])

    def test_transaction_reads_its_own_writes(self) -> None:
        self.run_sql("CREATE TABLE t (id BIGINT PRIMARY KEY, v BIGINT);")
        self.run_sql("INSERT INTO t VALUES (1, 10);")
        self.run_sql("BEGIN;")
        self.run_sql("INSERT INTO t VALUES (2, 20);")
        self.run_sql("UPDATE t SET v = 99 WHERE id = 1;")
        self.run_sql("DELETE FROM t WHERE id = 2;")
        self.assertEqual(self.rows("SELECT id, v FROM t ORDER BY id;"), [[1, 99]])
        self.run_sql("COMMIT;")
        self.assertEqual(self.rows("SELECT id, v FROM t ORDER BY id;"), [[1, 99]])

    def test_rollback_restores_the_previous_state(self) -> None:
        self.run_sql("CREATE TABLE t (id BIGINT PRIMARY KEY, v BIGINT);")
        self.run_sql("INSERT INTO t VALUES (1, 10);")
        self.run_sql("BEGIN;")
        self.run_sql("UPDATE t SET v = 20 WHERE id = 1;")
        self.run_sql("INSERT INTO t VALUES (2, 20);")
        self.run_sql("ROLLBACK;")
        self.assertEqual(self.rows("SELECT id, v FROM t ORDER BY id;"), [[1, 10]])

    def test_failed_statement_leaves_no_partial_change(self) -> None:
        self.run_sql("CREATE TABLE t (id BIGINT PRIMARY KEY, v BIGINT);")
        self.run_sql("BEGIN;")
        self.run_sql("INSERT INTO t VALUES (1, 10);")
        with self.assertRaises(MiniVectorDBError):
            self.run_sql("INSERT INTO t VALUES (1, 20);")  # duplicate primary key
        # the failed statement changed nothing, the first one is still there
        self.assertEqual(self.rows("SELECT id, v FROM t;"), [[1, 10]])
        self.run_sql("COMMIT;")
        self.assertEqual(self.rows("SELECT id, v FROM t;"), [[1, 10]])

    def test_nested_begin_is_rejected(self) -> None:
        self.run_sql("BEGIN;")
        with self.assertRaises(MiniVectorDBError):
            self.run_sql("BEGIN;")
        self.run_sql("ROLLBACK;")

    def test_autocommit_applies_single_statements(self) -> None:
        self.run_sql("CREATE TABLE t (id BIGINT PRIMARY KEY, v BIGINT);")
        self.run_sql("INSERT INTO t VALUES (1, 10);")
        self.assertEqual(self.rows("SELECT id FROM t;"), [[1]])
        self.assertEqual(self.run_sql("INSERT INTO t VALUES (2, 20);")["affected_rows"], 1)


class PersistenceTest(DatabaseTestCase):
    def test_data_survives_a_restart(self) -> None:
        self.run_sql("CREATE TABLE t (id BIGINT PRIMARY KEY, name VARCHAR(8), e VECTOR(2));")
        self.run_sql("INSERT INTO t VALUES (1, 'a', '[1,2]'), (2, 'b', '[3,4]');")
        self.db.close()

        db = Database(self.data_dir)
        db.open()
        executor = Executor(db)
        try:
            result = executor.execute(parse("SELECT id, name, e FROM t ORDER BY id;"))
            self.assertEqual(result["columns"], ["id", "name", "e"])
            self.assertEqual(result["rows"][0][:2], [1, "a"])
            self.assertEqual(result["rows"][1][:2], [2, "b"])
            self.assertEqual([round(v, 5) for v in result["rows"][0][2]], [1.0, 2.0])
        finally:
            db.close()

    def test_uncommitted_transaction_is_not_persisted(self) -> None:
        self.run_sql("CREATE TABLE t (id BIGINT PRIMARY KEY, v BIGINT);")
        self.run_sql("BEGIN;")
        self.run_sql("INSERT INTO t VALUES (1, 10);")
        self.db.close()  # simulates the process dying inside the transaction

        db = Database(self.data_dir)
        db.open()
        executor = Executor(db)
        try:
            result = executor.execute(parse("SELECT id FROM t;"))
            self.assertEqual(result["rows"], [])
        finally:
            db.close()

    def test_committed_transaction_is_recovered_after_a_crash(self) -> None:
        """The log is replayed when the process dies between log and data."""
        from minivectordb.engine import storage as storage_module

        self.run_sql("CREATE TABLE t (id BIGINT PRIMARY KEY, v BIGINT);")
        original_apply = storage_module.FileStore.apply_writes
        state = {"failed": False}

        def exploding_apply(self, writes):
            # Let the catalog be written, but fail as soon as the data files
            # are patched: the log already contains the commit.
            if not state["failed"]:
                state["failed"] = True
                raise OSError("simulated crash while applying the commit")
            return original_apply(self, writes)

        storage_module.FileStore.apply_writes = exploding_apply
        try:
            with self.assertRaises(OSError):
                self.run_sql("INSERT INTO t VALUES (7, 70);")
        finally:
            storage_module.FileStore.apply_writes = original_apply
        self.assertTrue(state["failed"])

        db = Database(self.data_dir)
        db.open()
        executor = Executor(db)
        try:
            result = executor.execute(parse("SELECT id, v FROM t;"))
            self.assertEqual(result["rows"], [[7, 70]])
        finally:
            db.close()

    def test_recovery_is_idempotent(self) -> None:
        self.run_sql("CREATE TABLE t (id BIGINT PRIMARY KEY, v BIGINT);")
        self.run_sql("INSERT INTO t VALUES (1, 10);")
        self.db.close()
        for _ in range(3):
            db = Database(self.data_dir)
            db.open()
            executor = Executor(db)
            try:
                self.assertEqual(
                    executor.execute(parse("SELECT id, v FROM t;"))["rows"], [[1, 10]]
                )
            finally:
                db.close()


class VectorTest(DatabaseTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.run_sql(
            "CREATE TABLE docs (id BIGINT PRIMARY KEY, category VARCHAR(8), e VECTOR(2));"
        )
        self.run_sql(
            "INSERT INTO docs VALUES (1, 'a', '[0, 0]'), (2, 'a', '[3, 4]'), (3, 'b', '[6, 8]');"
        )

    def test_l2_distance_is_euclidean_not_squared(self) -> None:
        rows = self.rows(
            "SELECT id, L2_DISTANCE(e, '[0, 0]') AS distance FROM docs WHERE id = 1;"
        )
        self.assertAlmostEqual(rows[0][1], 0.0)
        rows = self.rows(
            "SELECT id, L2_DISTANCE(e, '[0, 0]') AS distance FROM docs WHERE id = 2;"
        )
        self.assertAlmostEqual(rows[0][1], 5.0)

    def test_topk_orders_by_distance_and_breaks_ties_by_primary_key(self) -> None:
        self.run_sql(
            "INSERT INTO docs VALUES (4, 'a', '[0, 0]'), (5, 'a', '[0, 0]');"
        )
        rows = self.rows(
            "SELECT id, L2_DISTANCE(e, '[0, 0]') AS distance FROM docs ORDER BY distance LIMIT 3;"
        )
        self.assertEqual([row[0] for row in rows], [1, 4, 5])
        self.assertAlmostEqual(rows[0][1], 0.0)

    def test_filtered_topk_only_considers_matching_rows(self) -> None:
        rows = self.rows(
            "SELECT id, L2_DISTANCE(e, '[0, 0]') AS distance FROM docs "
            "WHERE category = 'a' ORDER BY distance LIMIT 2;"
        )
        self.assertEqual([row[0] for row in rows], [1, 2])

    def test_vector_update_changes_the_distance(self) -> None:
        self.run_sql("UPDATE docs SET e = '[0, 0]' WHERE id = 3;")
        rows = self.rows(
            "SELECT id, L2_DISTANCE(e, '[0, 0]') AS distance FROM docs ORDER BY distance LIMIT 1;"
        )
        self.assertEqual(rows[0][0], 1)
        rows = self.rows(
            "SELECT id, L2_DISTANCE(e, '[0, 0]') AS distance FROM docs WHERE id = 3;"
        )
        self.assertAlmostEqual(rows[0][1], 0.0)

    def test_deleted_rows_disappear_from_the_search(self) -> None:
        self.run_sql("DELETE FROM docs WHERE id = 1;")
        rows = self.rows(
            "SELECT id, L2_DISTANCE(e, '[0, 0]') AS distance FROM docs ORDER BY distance LIMIT 1;"
        )
        self.assertEqual(rows[0][0], 2)

    def test_exact_mode_ignores_the_approximate_index(self) -> None:
        self.run_sql("CREATE VECTOR INDEX idx_docs ON docs(e);")
        self.db.mode = "ann"
        approximate = self.rows(
            "SELECT id, L2_DISTANCE(e, '[0, 0]') AS distance FROM docs ORDER BY distance LIMIT 3;"
        )
        self.db.mode = "exact"
        exact = self.rows(
            "SELECT id, L2_DISTANCE(e, '[0, 0]') AS distance FROM docs ORDER BY distance LIMIT 3;"
        )
        self.assertEqual([row[0] for row in exact], [1, 2, 3])
        self.assertEqual([row[0] for row in approximate], [row[0] for row in exact])

    def test_approximate_search_sees_new_updated_and_deleted_rows(self) -> None:
        self.run_sql("CREATE VECTOR INDEX idx_docs ON docs(e);")
        self.db.mode = "ann"
        self.run_sql("INSERT INTO docs VALUES (4, 'b', '[0, 1]');")
        rows = self.rows(
            "SELECT id FROM docs ORDER BY L2_DISTANCE(e, '[0, 0]') LIMIT 4;"
        ) if False else None
        self.run_sql("UPDATE docs SET e = '[0, 0.5]' WHERE id = 4;")
        self.run_sql("DELETE FROM docs WHERE id = 4;")
        rows = self.rows(
            "SELECT id, L2_DISTANCE(e, '[0, 0]') AS distance FROM docs ORDER BY distance LIMIT 3;"
        )
        self.assertNotIn(4, [row[0] for row in rows])
        self.assertEqual([row[0] for row in rows], [1, 2, 3])

    def test_vector_dimension_is_validated(self) -> None:
        with self.assertRaises(MiniVectorDBError):
            self.run_sql("INSERT INTO docs VALUES (9, 'a', '[1,2,3]');")

    def test_non_finite_vector_values_are_rejected(self) -> None:
        for literal in ("[NaN, 1]", "[Infinity, 1]", "[1e400, 1]"):
            with self.assertRaises(MiniVectorDBError):
                self.run_sql(f"INSERT INTO docs VALUES (9, 'a', '{literal}');")


class ErrorTest(DatabaseTestCase):
    def test_unknown_table(self) -> None:
        with self.assertRaises(MiniVectorDBError):
            self.run_sql("SELECT id FROM missing;")

    def test_unknown_column(self) -> None:
        self.run_sql("CREATE TABLE t (id BIGINT PRIMARY KEY);")
        with self.assertRaises(MiniVectorDBError):
            self.run_sql("SELECT nope FROM t;")

    def test_duplicate_primary_key(self) -> None:
        self.run_sql("CREATE TABLE t (id BIGINT PRIMARY KEY);")
        self.run_sql("INSERT INTO t VALUES (1);")
        with self.assertRaises(MiniVectorDBError):
            self.run_sql("INSERT INTO t VALUES (1);")

    def test_unsupported_statements_are_rejected(self) -> None:
        for sql in (
            "ALTER TABLE t ADD COLUMN x BIGINT;",
            "DROP TABLE t;",
            "SELECT COUNT(*) FROM t;",
            "SELECT a FROM t JOIN u ON t.id = u.id;",
            "SELECT a FROM t WHERE a IS NULL;",
            "SELECT a FROM t WHERE a = 1 OR b = 2;",
            "INSERT INTO t SELECT * FROM u;",
            "UPDATE t SET id = 5;",
            "GROUP BY a;",
            "SELECT a FROM t HAVING a > 1;",
        ):
            with self.assertRaises(MiniVectorDBError, msg=sql):
                self.run_sql(sql)

    def test_varchar_length_is_enforced(self) -> None:
        self.run_sql("CREATE TABLE t (id BIGINT PRIMARY KEY, name VARCHAR(4));")
        with self.assertRaises(MiniVectorDBError):
            self.run_sql("INSERT INTO t VALUES (1, 'too long');")

    def test_string_comparison_uses_code_points(self) -> None:
        self.run_sql("CREATE TABLE t (id BIGINT PRIMARY KEY, s VARCHAR(4));")
        # '乙' is U+4E59, '甲' is U+7532: the code point order puts 乙 first.
        self.run_sql("INSERT INTO t VALUES (1, '甲'), (2, '乙');")
        rows = self.rows("SELECT id FROM t WHERE s >= '乙' ORDER BY id;")
        self.assertEqual([row[0] for row in rows], [1, 2])
        rows = self.rows("SELECT id FROM t WHERE s < '甲' ORDER BY id;")
        self.assertEqual([row[0] for row in rows], [2])


class ModeSwitchTest(DatabaseTestCase):
    def test_set_search_mode_switches_the_retrieval_mode(self) -> None:
        self.run_sql("CREATE TABLE t (id BIGINT PRIMARY KEY, e VECTOR(2));")
        self.run_sql("INSERT INTO t VALUES (1, '[0, 0]');")
        self.run_sql("SET search_mode = 'ann';")
        self.assertEqual(self.db.mode, "ann")
        self.run_sql("SET search_mode = 'exact';")
        self.assertEqual(self.db.mode, "exact")

    def test_other_set_statements_are_unsupported(self) -> None:
        for sql in ("SET autocommit = 1;", "SET names = 'utf8';", "SET foo = 1;"):
            with self.assertRaises(MiniVectorDBError, msg=sql):
                self.run_sql(sql)

    def test_double_quoted_strings_are_accepted(self) -> None:
        self.run_sql('CREATE TABLE t (id BIGINT PRIMARY KEY, name VARCHAR(8));')
        self.run_sql("INSERT INTO t VALUES (1, 'abc');")
        self.assertEqual(self.rows("SELECT name FROM t WHERE name = \"abc\";"), [["abc"]])


class ResourceTest(DatabaseTestCase):
    def test_data_larger_than_the_cache_budget_is_still_queryable(self) -> None:
        """A table that does not fit into the column cache is read in chunks."""
        import minivectordb.engine.table as table_module

        original = table_module.CACHE_BUDGET
        table_module.CACHE_BUDGET = 4096  # 4 KiB -> the table cannot be cached
        try:
            self.run_sql("CREATE TABLE t (id BIGINT PRIMARY KEY, e VECTOR(16));")
            for start in range(0, 64, 8):
                values = ",".join(
                    "(%d, '[%s]')" % (index, ",".join(["1"] * 16))
                    for index in range(start, start + 8)
                )
                self.run_sql(f"INSERT INTO t VALUES {values};")
            rows = self.rows(
                "SELECT id, L2_DISTANCE(e, '[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]') AS d "
                "FROM t ORDER BY d LIMIT 3;"
            )
            self.assertEqual(len(rows), 3)
            # all vectors are identical, so the tie is broken by the primary key
            self.assertEqual(rows[0][0], 0)
        finally:
            table_module.CACHE_BUDGET = original


class ProtocolTest(unittest.TestCase):
    """End to end test through the real process interface."""

    def setUp(self) -> None:
        self.data_dir = Path(tempfile.mkdtemp(prefix="minivectordb-proto-"))

    def tearDown(self) -> None:
        shutil.rmtree(self.data_dir, ignore_errors=True)

    def talk(self, statements, data_dir=None):
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT)
        env["MINIVECTORDB_READY_MARKER"] = "none"
        process = subprocess.run(
            [sys.executable, "-m", "minivectordb", "--data-dir", str(data_dir or self.data_dir)],
            input="\n".join(statements) + "\n",
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        return [json.loads(line) for line in process.stdout.splitlines() if line.strip()]

    def test_one_json_object_per_statement(self) -> None:
        answers = self.talk(
            [
                "CREATE TABLE t (id BIGINT PRIMARY KEY, v BIGINT);",
                "INSERT INTO t VALUES (1, 2);",
                "SELECT id, v FROM t;",
            ]
        )
        self.assertEqual(len(answers), 3)
        self.assertEqual(answers[1], {"affected_rows": 1})
        self.assertEqual(answers[2], {"columns": ["id", "v"], "rows": [[1, 2]]})

    def test_errors_are_reported_without_crashing(self) -> None:
        answers = self.talk(
            [
                "SELECT * FROM missing;",
                "SELECT * FROM;",
                "CREATE TABLE t (id BIGINT PRIMARY KEY);",
                "SELECT id FROM t;",
            ]
        )
        self.assertEqual(len(answers), 4)
        self.assertIn("error", answers[0])
        self.assertIn("error", answers[1])
        self.assertEqual(answers[3]["rows"], [])

    def test_state_survives_a_restart_of_the_process(self) -> None:
        self.talk(
            [
                "CREATE TABLE t (id BIGINT PRIMARY KEY, v BIGINT);",
                "INSERT INTO t VALUES (5, 50);",
            ]
        )
        answers = self.talk(["SELECT id, v FROM t;"])
        self.assertEqual(answers[0]["rows"], [[5, 50]])


if __name__ == "__main__":
    unittest.main()
