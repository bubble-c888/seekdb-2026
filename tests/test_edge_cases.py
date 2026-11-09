"""Edge cases of the formal requirements.

The public mysqltest samples only cover a part of the task statement.  The
tests in this module check the remaining requirements that the formal cases
are built from:

* the error semantics of the task package (``duplicate_key``,
  ``unsupported_sql``, ``dimension_mismatch``, ...);
* statements that must fail (primary key updates, NULL values, unsupported
  constructs);
* the string semantics (Unicode length, code point order, a value that is not
  part of the dictionary);
* the approximate mode on a data set that is large enough to use the index
  instead of the exact scan;
* the crash recovery of a process that was killed instead of shut down.

Run them with::

    python -m unittest tests.test_edge_cases -v
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


class EngineTestCase(unittest.TestCase):
    """Base class that gives every test its own empty data directory."""

    def setUp(self) -> None:
        self.data_dir = Path(tempfile.mkdtemp(prefix="minivectordb-edge-"))
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

    def fails(self, sql: str) -> str:
        """Run a statement that has to fail and return the error name."""
        with self.assertRaises(MiniVectorDBError) as caught:
            self.executor.execute(parse(sql))
        return caught.exception.name


class ErrorSemanticsTest(EngineTestCase):
    """The error names of the task package."""

    def test_duplicate_key(self) -> None:
        self.run_sql("CREATE TABLE t (id BIGINT PRIMARY KEY);")
        self.run_sql("INSERT INTO t VALUES (1);")
        self.assertEqual(self.fails("INSERT INTO t VALUES (1);"), "duplicate_key")

    def test_unsupported_sql(self) -> None:
        self.run_sql("CREATE TABLE t (id BIGINT PRIMARY KEY);")
        self.assertEqual(self.fails("ALTER TABLE t ADD x BIGINT;"), "unsupported_sql")
        self.assertEqual(self.fails("DROP TABLE t;"), "unsupported_sql")
        self.assertEqual(self.fails("DELETE FROM t WHERE id IN (1, 2);"), "unsupported_sql")
        self.assertEqual(self.fails("SELECT COUNT(*) FROM t;"), "unsupported_sql")
        self.assertEqual(
            self.fails("SELECT id FROM t WHERE id IN (SELECT id FROM t);"),
            "unsupported_sql",
        )
        self.assertEqual(
            self.fails("SELECT a.id FROM t JOIN t AS a ON t.id = a.id;"),
            "unsupported_sql",
        )

    def test_dimension_mismatch(self) -> None:
        self.run_sql("CREATE TABLE t (id BIGINT PRIMARY KEY, e VECTOR(2));")
        self.assertEqual(
            self.fails("INSERT INTO t VALUES (1, '[1,2,3]');"), "dimension_mismatch"
        )
        self.assertEqual(
            self.fails("INSERT INTO t VALUES (1, '[1]');"), "dimension_mismatch"
        )

    def test_syntax_error(self) -> None:
        self.assertEqual(self.fails("SELCT id FROM t;"), "syntax_error")

    def test_error_response_carries_the_name_only(self) -> None:
        from minivectordb.errors import DuplicateKeyError

        error = DuplicateKeyError("boom")
        self.assertEqual(error.to_dict(), {"error": "duplicate_key"})
        log = error.to_log()
        self.assertEqual(log["error"], "duplicate_key")
        self.assertEqual(log["code"], 1062)
        self.assertEqual(log["sqlstate"], "23000")


class RejectedStatementTest(EngineTestCase):
    """Statements that the task statement forbids."""

    def test_primary_key_update_is_rejected(self) -> None:
        self.run_sql("CREATE TABLE t (id BIGINT PRIMARY KEY, v BIGINT);")
        self.run_sql("INSERT INTO t VALUES (1, 10);")
        name = self.fails("UPDATE t SET id = 2 WHERE id = 1;")
        self.assertIn(name, ("unsupported_sql", "transaction_state_error"))
        self.assertEqual(self.rows("SELECT id, v FROM t;"), [[1, 10]])

    def test_null_values_are_rejected(self) -> None:
        self.run_sql("CREATE TABLE t (id BIGINT PRIMARY KEY, v BIGINT);")
        self.assertTrue(self.fails("INSERT INTO t VALUES (1, NULL);"))
        self.assertEqual(self.rows("SELECT id FROM t;"), [])

    def test_wrong_number_of_values_is_rejected(self) -> None:
        self.run_sql("CREATE TABLE t (id BIGINT PRIMARY KEY, v BIGINT);")
        self.assertTrue(self.fails("INSERT INTO t VALUES (1);"))
        self.assertTrue(self.fails("INSERT INTO t VALUES (1, 2, 3);"))
        self.assertEqual(self.rows("SELECT id FROM t;"), [])

    def test_second_explicit_transaction_is_rejected(self) -> None:
        self.run_sql("BEGIN;")
        self.assertTrue(self.fails("BEGIN;"))
        self.run_sql("ROLLBACK;")

    def test_commit_without_a_transaction_is_accepted(self) -> None:
        # MySQL treats a COMMIT without an open transaction as a no-op.
        self.run_sql("COMMIT;")

    def test_table_with_two_primary_keys_is_rejected(self) -> None:
        self.assertTrue(
            self.fails(
                "CREATE TABLE t (id BIGINT PRIMARY KEY, other BIGINT PRIMARY KEY);"
            )
        )

    def test_two_vector_columns_are_rejected(self) -> None:
        self.assertTrue(
            self.fails(
                "CREATE TABLE t (id BIGINT PRIMARY KEY, a VECTOR(2), b VECTOR(2));"
            )
        )


class StringSemanticsTest(EngineTestCase):
    """VARCHAR and string comparison."""

    def test_length_is_counted_in_unicode_characters(self) -> None:
        self.run_sql("CREATE TABLE t (id BIGINT PRIMARY KEY, s VARCHAR(2));")
        self.run_sql("INSERT INTO t VALUES (1, '中文');")
        self.assertEqual(self.rows("SELECT s FROM t;"), [["中文"]])
        self.assertTrue(self.fails("INSERT INTO t VALUES (2, '中文文');"))

    def test_equality_with_an_unknown_value_matches_nothing(self) -> None:
        """A value that is not part of the dictionary must not match a row."""
        self.run_sql("CREATE TABLE t (id BIGINT PRIMARY KEY, s VARCHAR(8));")
        self.run_sql("INSERT INTO t VALUES (1, 'keep'), (2, 'skip');")
        self.assertEqual(self.rows("SELECT id FROM t WHERE s = 'keep';"), [[1]])
        self.assertEqual(self.rows("SELECT id FROM t WHERE s = 'absent';"), [])
        self.assertEqual(self.rows("SELECT id FROM t WHERE s = 'zzz';"), [])

    def test_equality_with_an_unknown_value_matches_nothing_in_a_transaction(self) -> None:
        self.run_sql("CREATE TABLE t (id BIGINT PRIMARY KEY, s VARCHAR(8));")
        self.run_sql("INSERT INTO t VALUES (1, 'keep');")
        self.run_sql("BEGIN;")
        self.run_sql("INSERT INTO t VALUES (2, 'other');")
        self.assertEqual(self.rows("SELECT id FROM t WHERE s = 'absent';"), [])
        self.assertEqual(self.rows("SELECT id FROM t WHERE s = 'other';"), [[2]])
        self.run_sql("ROLLBACK;")

    def test_ordering_uses_code_points(self) -> None:
        self.run_sql("CREATE TABLE t (id BIGINT PRIMARY KEY, s VARCHAR(8));")
        self.run_sql("INSERT INTO t VALUES (1, '甲'), (2, '乙');")
        self.assertEqual(self.rows("SELECT id FROM t ORDER BY s;"), [[2], [1]])

    def test_ordering_comparison_with_an_unknown_value(self) -> None:
        self.run_sql("CREATE TABLE t (id BIGINT PRIMARY KEY, s VARCHAR(8));")
        self.run_sql("INSERT INTO t VALUES (1, 'aa'), (2, 'bb');")
        self.assertEqual(self.rows("SELECT id FROM t WHERE s < 'ab';"), [[1]])
        self.assertEqual(self.rows("SELECT id FROM t WHERE s >= 'ab';"), [[2]])

    def test_empty_string_is_a_value(self) -> None:
        self.run_sql("CREATE TABLE t (id BIGINT PRIMARY KEY, s VARCHAR(8));")
        self.run_sql("INSERT INTO t VALUES (1, '');")
        self.assertEqual(self.rows("SELECT id FROM t WHERE s = '';"), [[1]])


class OrderingTest(EngineTestCase):
    """Ties are broken by the primary key."""

    def test_order_by_ties_use_the_primary_key(self) -> None:
        self.run_sql("CREATE TABLE t (id BIGINT PRIMARY KEY, v BIGINT);")
        self.run_sql("INSERT INTO t VALUES (3, 1), (1, 1), (2, 1);")
        # The task statement breaks ties with the primary key in ascending
        # order, also for a descending sort.
        self.assertEqual(self.rows("SELECT id FROM t ORDER BY v;"), [[1], [2], [3]])
        self.assertEqual(self.rows("SELECT id FROM t ORDER BY v DESC;"), [[1], [2], [3]])

    def test_limit_zero_returns_nothing(self) -> None:
        self.run_sql("CREATE TABLE t (id BIGINT PRIMARY KEY, v BIGINT);")
        self.run_sql("INSERT INTO t VALUES (1, 1);")
        self.assertEqual(self.rows("SELECT id FROM t LIMIT 0;"), [])

    def test_topk_ties_use_the_primary_key(self) -> None:
        self.run_sql("CREATE TABLE t (id BIGINT PRIMARY KEY, e VECTOR(2));")
        self.run_sql("INSERT INTO t VALUES (5, '[0,0]'), (2, '[0,0]'), (9, '[0,0]');")
        rows = self.rows(
            "SELECT id, L2_DISTANCE(e, '[0,0]') AS distance FROM t ORDER BY distance LIMIT 2;"
        )
        self.assertEqual([row[0] for row in rows], [2, 5])


class ApproximateModeTest(EngineTestCase):
    """The approximate mode with a data set that uses the index."""

    def setUp(self) -> None:
        super().setUp()
        os.environ["MINIVECTORDB_ANN_EXACT_LIMIT"] = "0"
        import minivectordb.engine.executor as executor_module

        self._limit = executor_module.ANN_EXACT_LIMIT
        executor_module.ANN_EXACT_LIMIT = 0

    def tearDown(self) -> None:
        import minivectordb.engine.executor as executor_module

        executor_module.ANN_EXACT_LIMIT = self._limit
        os.environ.pop("MINIVECTORDB_ANN_EXACT_LIMIT", None)
        super().tearDown()

    def _table(self, count: int = 400):
        self.run_sql("CREATE TABLE t (id BIGINT PRIMARY KEY, g VARCHAR(8), e VECTOR(2));")
        values = []
        for index in range(count):
            angle = index * 0.05
            values.append(
                "({}, 'g{}', '[{:.4f},{:.4f}]')".format(
                    index + 1, index % 4, round(1.0 * index / count, 6), round(index % 7, 6)
                )
            )
        self.run_sql("INSERT INTO t VALUES " + ", ".join(values) + ";")

    def test_approximate_results_match_the_exact_results(self) -> None:
        self._table()
        self.run_sql("CREATE VECTOR INDEX idx_t ON t(e);")
        query = (
            "SELECT id, L2_DISTANCE(e, '[0.5,0.25]') AS distance FROM t "
            "ORDER BY distance LIMIT 10;"
        )
        exact = self.rows(query)
        self.db.mode = "ann"
        approximate = self.rows(query)
        self.assertEqual(
            [row[0] for row in approximate], [row[0] for row in exact]
        )

    def test_approximate_search_respects_the_filter(self) -> None:
        self._table()
        self.run_sql("CREATE VECTOR INDEX idx_t ON t(e);")
        self.db.mode = "ann"
        rows = self.rows(
            "SELECT id FROM t WHERE g = 'g1' ORDER BY L2_DISTANCE(e, '[0,0]') LIMIT 5;"
        )
        self.assertTrue(rows)
        for row in rows:
            # the group of a row is (id - 1) % 4
            self.assertEqual((row[0] - 1) % 4, 1)

    def test_approximate_recall_is_high(self) -> None:
        import random

        self._table()
        self.run_sql("CREATE VECTOR INDEX idx_t ON t(e);")
        random.seed(7)
        hits = 0
        total = 0
        for _ in range(20):
            query = "[{:.4f},{:.4f}]".format(random.random(), random.random())
            exact = [
                row[0]
                for row in self.rows(
                    "SELECT id, L2_DISTANCE(e, '" + query + "') AS distance FROM t "
                    "ORDER BY distance LIMIT 10;"
                )
            ]
            self.db.mode = "ann"
            approximate = [
                row[0]
                for row in self.rows(
                    "SELECT id, L2_DISTANCE(e, '" + query + "') AS distance FROM t "
                    "ORDER BY distance LIMIT 10;"
                )
            ]
            self.db.mode = "exact"
            hits += len(set(exact) & set(approximate))
            total += len(exact)
        self.assertGreaterEqual(hits / float(total), 0.9)

    def test_approximate_search_returns_no_duplicates(self) -> None:
        self._table()
        self.run_sql("CREATE VECTOR INDEX idx_t ON t(e);")
        self.db.mode = "ann"
        rows = self.rows(
            "SELECT id, L2_DISTANCE(e, '[0,0]') AS distance FROM t ORDER BY distance LIMIT 25;"
        )
        ids = [row[0] for row in rows]
        self.assertEqual(len(ids), len(set(ids)))

    def test_approximate_search_only_returns_live_rows(self) -> None:
        self._table()
        self.run_sql("CREATE VECTOR INDEX idx_t ON t(e);")
        self.run_sql("DELETE FROM t WHERE id <= 10;")
        self.db.mode = "ann"
        rows = self.rows(
            "SELECT id FROM t ORDER BY L2_DISTANCE(e, '[0,0]') LIMIT 50;"
        )
        for row in rows:
            self.assertGreater(row[0], 10)


class CrashRecoveryTest(unittest.TestCase):
    """A process that is killed keeps the committed data."""

    def setUp(self) -> None:
        self.data_dir = Path(tempfile.mkdtemp(prefix="minivectordb-crash-"))

    def tearDown(self) -> None:
        shutil.rmtree(self.data_dir, ignore_errors=True)

    def _program(self) -> str:
        return "{} -m minivectordb --data-dir {}".format(sys.executable, self.data_dir)

    def test_committed_data_survives_a_kill(self) -> None:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(REPO_ROOT)
        environment["PYTHONIOENCODING"] = "utf-8"
        environment["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            [sys.executable, "-m", "minivectordb", "--data-dir", str(self.data_dir)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env=environment,
            cwd=str(REPO_ROOT),
        )
        try:
            process.stdin.write(
                "CREATE TABLE t (id BIGINT PRIMARY KEY, v BIGINT);\n"
                "INSERT INTO t VALUES (1, 10);\n"
            )
            process.stdin.flush()
            answers = [process.stdout.readline() for _ in range(2)]
            self.assertEqual(json.loads(answers[1]), {"affected_rows": 1})
            # The commit is durable: the process dies without any shutdown.
            process.kill()
            process.wait(timeout=30)
        finally:
            if process.poll() is None:  # pragma: no cover - defensive
                process.kill()

        database = Database(self.data_dir)
        try:
            database.open()
            executor = Executor(database)
            rows = executor.execute(parse("SELECT id, v FROM t;"))["rows"]
            self.assertEqual(rows, [[1, 10]])
        finally:
            database.close()

    def test_uncommitted_data_does_not_survive_a_kill(self) -> None:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(REPO_ROOT)
        environment["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            [sys.executable, "-m", "minivectordb", "--data-dir", str(self.data_dir)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env=environment,
            cwd=str(REPO_ROOT),
        )
        try:
            process.stdin.write(
                "CREATE TABLE t (id BIGINT PRIMARY KEY, v BIGINT);\n"
                "INSERT INTO t VALUES (1, 10);\n"
                "BEGIN;\n"
                "INSERT INTO t VALUES (2, 20);\n"
            )
            process.stdin.flush()
            for _ in range(4):
                process.stdout.readline()
            process.kill()
            process.wait(timeout=30)
        finally:
            if process.poll() is None:  # pragma: no cover - defensive
                process.kill()

        database = Database(self.data_dir)
        try:
            database.open()
            executor = Executor(database)
            rows = executor.execute(parse("SELECT id FROM t;"))["rows"]
            self.assertEqual(rows, [[1]])
        finally:
            database.close()


class ProtocolTest(unittest.TestCase):
    """The stdin/stdout envelope."""

    def setUp(self) -> None:
        self.data_dir = Path(tempfile.mkdtemp(prefix="minivectordb-protocol-"))

    def tearDown(self) -> None:
        shutil.rmtree(self.data_dir, ignore_errors=True)

    def _talk(self, payload: str, newline_at_the_end: bool = True) -> list:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(REPO_ROOT)
        environment["PYTHONIOENCODING"] = "utf-8"
        environment["PYTHONUNBUFFERED"] = "1"
        process = subprocess.run(
            [sys.executable, "-m", "minivectordb", "--data-dir", str(self.data_dir)],
            input=payload,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env=environment,
            cwd=str(REPO_ROOT),
            timeout=120,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        return [json.loads(line) for line in process.stdout.splitlines() if line.strip()]

    def test_a_statement_without_a_newline_is_answered(self) -> None:
        answers = self._talk("SELECT 1;")
        self.assertEqual(len(answers), 1)

    def test_a_statement_without_a_semicolon_is_answered(self) -> None:
        answers = self._talk("CREATE TABLE t (id BIGINT PRIMARY KEY);\nSELECT id FROM t")
        self.assertEqual(len(answers), 2)
        self.assertEqual(answers[1], {"columns": ["id"], "rows": []})

    def test_a_directive_is_not_answered(self) -> None:
        answers = self._talk(
            "CREATE TABLE t (id BIGINT PRIMARY KEY);\n--mode ann\nSELECT id FROM t;\n"
        )
        self.assertEqual(len(answers), 2)

    def test_a_comment_is_not_answered(self) -> None:
        answers = self._talk("-- a comment\nSELECT 1;\n")
        self.assertEqual(len(answers), 1)

    def test_stdout_carries_only_results(self) -> None:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(REPO_ROOT)
        environment["PYTHONUNBUFFERED"] = "1"
        process = subprocess.run(
            [sys.executable, "-m", "minivectordb", "--data-dir", str(self.data_dir)],
            input="SELECT 1;\n",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env=environment,
            cwd=str(REPO_ROOT),
            timeout=120,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        lines = [line for line in process.stdout.splitlines() if line.strip()]
        self.assertEqual(len(lines), 1)
        json.loads(lines[0])


if __name__ == "__main__":
    unittest.main()
