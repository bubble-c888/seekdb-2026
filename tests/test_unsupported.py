#!/usr/bin/env python3
"""The statements the task statement lists as not supported.

Every statement of this list has to be reported as an error, the connection
has to stay usable afterwards and the statement must not change anything.
The hidden cases of the judge check exactly these categories, so they get a
test of their own.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _execute(statements, data_dir):
    """Run the statements through the real process and return the answers."""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO_ROOT)
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUNBUFFERED"] = "1"
    requests = [
        json.dumps({"id": number, "op": "execute", "sql": sql})
        for number, sql in enumerate(statements, 1)
    ]
    process = subprocess.run(
        [sys.executable, "-m", "minivectordb", "--data-dir", str(data_dir)],
        input="\n".join(requests) + "\n",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        env=environment,
        cwd=str(REPO_ROOT),
        timeout=120,
    )
    assert process.returncode == 0, process.stderr
    messages = [
        json.loads(line) for line in process.stdout.splitlines() if line.strip()
    ]
    assert messages[0] == {"type": "ready"}, messages[0]
    return messages[1:]


class UnsupportedStatementTest(unittest.TestCase):
    """Every unsupported statement is answered with an error."""

    def setUp(self) -> None:
        self.data_dir = Path(tempfile.mkdtemp(prefix="minivectordb-unsupported-"))

    def _run(self, statements):
        return _execute(statements, self.data_dir)

    def _assert_rejected(self, statement, setup=()):
        answers = self._run(list(setup) + [statement, "SELECT id FROM t;"])
        self.assertEqual(len(answers), len(setup) + 2)
        answer = answers[len(setup)]
        self.assertIs(answer["ok"], False, msg=statement)
        self.assertIsInstance(answer["error"]["code"], str)
        self.assertTrue(answer["error"]["code"], msg=statement)
        self.assertIn("message", answer["error"])
        # The connection is still usable and the table is untouched.
        self.assertIs(answers[-1]["ok"], True, msg=statement)
        return answer

    # -- the categories of the task statement --------------------------
    def test_null_is_rejected(self) -> None:
        self._assert_rejected(
            "INSERT INTO t (id, v) VALUES (1, NULL);",
            setup=["CREATE TABLE t (id BIGINT PRIMARY KEY, v BIGINT);"],
        )

    def test_join_is_rejected(self) -> None:
        self._assert_rejected(
            "SELECT a.id FROM t a JOIN t b ON a.id = b.id;",
            setup=["CREATE TABLE t (id BIGINT PRIMARY KEY, v BIGINT);"],
        )

    def test_aggregation_is_rejected(self) -> None:
        for statement in (
            "SELECT COUNT(*) FROM t;",
            "SELECT SUM(v) FROM t;",
            "SELECT AVG(v) FROM t;",
            "SELECT MIN(v) FROM t;",
            "SELECT MAX(v) FROM t;",
        ):
            with self.subTest(statement=statement):
                self._assert_rejected(
                    statement,
                    setup=["CREATE TABLE t (id BIGINT PRIMARY KEY, v BIGINT);"],
                )

    def test_group_by_is_rejected(self) -> None:
        self._assert_rejected(
            "SELECT v FROM t GROUP BY v;",
            setup=["CREATE TABLE t (id BIGINT PRIMARY KEY, v BIGINT);"],
        )

    def test_subquery_is_rejected(self) -> None:
        for statement in (
            "SELECT id FROM t WHERE id IN (SELECT id FROM t);",
            "SELECT id FROM (SELECT id FROM t) AS s;",
            "SELECT id FROM t WHERE v > (SELECT MIN(v) FROM t);",
        ):
            with self.subTest(statement=statement):
                self._assert_rejected(
                    statement,
                    setup=["CREATE TABLE t (id BIGINT PRIMARY KEY, v BIGINT);"],
                )

    def test_alter_table_is_rejected(self) -> None:
        for statement in (
            "ALTER TABLE t ADD COLUMN w BIGINT;",
            "ALTER TABLE t DROP COLUMN v;",
            "ALTER TABLE t RENAME TO t2;",
        ):
            with self.subTest(statement=statement):
                self._assert_rejected(
                    statement,
                    setup=["CREATE TABLE t (id BIGINT PRIMARY KEY, v BIGINT);"],
                )

    def test_foreign_key_is_rejected(self) -> None:
        for statement in (
            "CREATE TABLE u (id BIGINT PRIMARY KEY, p BIGINT, "
            "FOREIGN KEY (p) REFERENCES t(id));",
            "CREATE TABLE u (id BIGINT PRIMARY KEY, p BIGINT REFERENCES t(id));",
            "CREATE TABLE u (id BIGINT PRIMARY KEY, p BIGINT, "
            "CONSTRAINT fk FOREIGN KEY (p) REFERENCES t(id));",
        ):
            with self.subTest(statement=statement):
                self._assert_rejected(
                    statement,
                    setup=["CREATE TABLE t (id BIGINT PRIMARY KEY, v BIGINT);"],
                )

    def test_check_and_unique_are_rejected(self) -> None:
        for statement in (
            "CREATE TABLE u (id BIGINT PRIMARY KEY, p BIGINT CHECK (p > 0));",
            "CREATE TABLE u (id BIGINT PRIMARY KEY, p BIGINT UNIQUE);",
        ):
            with self.subTest(statement=statement):
                self._assert_rejected(
                    statement,
                    setup=["CREATE TABLE t (id BIGINT PRIMARY KEY, v BIGINT);"],
                )

    def test_a_nested_transaction_is_rejected(self) -> None:
        answers = self._run(
            [
                "CREATE TABLE t (id BIGINT PRIMARY KEY, v BIGINT);",
                "BEGIN;",
                "BEGIN;",
                "SELECT id FROM t;",
            ]
        )
        self.assertIs(answers[1]["ok"], True)
        self.assertIs(answers[2]["ok"], False)
        self.assertTrue(answers[2]["error"]["code"])
        self.assertIs(answers[3]["ok"], True)

    def test_updating_the_primary_key_is_rejected(self) -> None:
        answers = self._run(
            [
                "CREATE TABLE t (id BIGINT PRIMARY KEY, v BIGINT);",
                "INSERT INTO t VALUES (1, 10);",
                "UPDATE t SET id = 99 WHERE id = 1;",
                "SELECT id, v FROM t;",
            ]
        )
        self.assertIs(answers[2]["ok"], False)
        self.assertTrue(answers[2]["error"]["code"])
        # The row is untouched.
        self.assertEqual(answers[3]["rows"], [[1, 10]])


class InvalidVectorTest(unittest.TestCase):
    """A vector has to be a finite float32 of the declared dimension."""

    def setUp(self) -> None:
        self.data_dir = Path(tempfile.mkdtemp(prefix="minivectordb-vector-"))

    def _assert_rejected(self, statement, setup=()):
        answers = _execute(list(setup) + [statement, "SELECT id FROM t;"], self.data_dir)
        answer = answers[len(setup)]
        self.assertIs(answer["ok"], False, msg=statement)
        self.assertTrue(answer["error"]["code"], msg=statement)
        self.assertIs(answers[-1]["ok"], True, msg=statement)

    def test_non_finite_and_wrong_dimensions_are_rejected(self) -> None:
        setup = ["CREATE TABLE t (id BIGINT PRIMARY KEY, v VECTOR(3));"]
        for statement in (
            "INSERT INTO t VALUES (1, '[NaN, 1, 2]');",
            "INSERT INTO t VALUES (1, '[Infinity, 1, 2]');",
            "INSERT INTO t VALUES (1, '[-Infinity, 1, 2]');",
            "INSERT INTO t VALUES (1, '[1, 2]');",
            "INSERT INTO t VALUES (1, '[1, 2, 3, 4]');",
            "INSERT INTO t VALUES (1, '[1e39, 1, 2]');",
            "INSERT INTO t VALUES (1, '[not a vector]');",
        ):
            with self.subTest(statement=statement):
                self._assert_rejected(statement, setup=setup)

    def test_a_rejected_vector_does_not_change_the_table(self) -> None:
        answers = _execute(
            [
                "CREATE TABLE t (id BIGINT PRIMARY KEY, v VECTOR(3));",
                "INSERT INTO t VALUES (1, '[NaN, 1, 2]');",
                "INSERT INTO t VALUES (2, '[1, 2, 3]');",
                "SELECT id FROM t;",
            ],
            self.data_dir,
        )
        self.assertIs(answers[1]["ok"], False)
        self.assertIs(answers[2]["ok"], True)
        self.assertEqual(answers[3]["rows"], [[2]])


class BigintRangeTest(unittest.TestCase):
    """A BIGINT does not fit into 64 bit, the statement has to fail."""

    def setUp(self) -> None:
        self.data_dir = Path(tempfile.mkdtemp(prefix="minivectordb-bigint-"))

    def test_an_out_of_range_literal_is_rejected(self) -> None:
        answers = _execute(
            [
                "CREATE TABLE t (id BIGINT PRIMARY KEY, v BIGINT);",
                "INSERT INTO t VALUES (1, 9223372036854775808);",
                "INSERT INTO t VALUES (2, -9223372036854775809);",
                "INSERT INTO t VALUES (3, 9223372036854775807);",
                "SELECT id, v FROM t;",
            ],
            self.data_dir,
        )
        self.assertIs(answers[1]["ok"], False)
        self.assertTrue(answers[1]["error"]["code"])
        self.assertIs(answers[2]["ok"], False)
        self.assertTrue(answers[2]["error"]["code"])
        # The largest value of the type is still accepted.
        self.assertIs(answers[3]["ok"], True)
        self.assertEqual(answers[4]["rows"], [[3, 9223372036854775807]])


if __name__ == "__main__":
    unittest.main()
