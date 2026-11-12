#!/usr/bin/env python3
"""Random queries against a naive reference implementation.

The public cases are a sample of the subset.  This test builds random
tables, applies random changes and compares the answers of the engine with
the answers of a reference that is written as plainly as possible: the rows
are a list of tuples and every query is evaluated with Python loops.  A
difference is a semantic difference, not a performance one.
"""

from __future__ import annotations

import math
import random
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from minivectordb.engine.database import Database  # noqa: E402
from minivectordb.engine.executor import Executor  # noqa: E402
from minivectordb.sqlparse import parse  # noqa: E402

DIM = 3
TOLERANCE = 1e-5
TEXTS = ("alpha", "beta", "gamma", "delta")


def float32(value: float) -> float:
    """The value as a float32 keeps it."""
    return float(np.float32(value))


class Reference:
    """The rows of one table, evaluated with plain Python."""

    def __init__(self) -> None:
        # every row: (id, g, s, vector)
        self.rows: Dict[int, Tuple[Any, ...]] = {}

    def insert(self, values) -> None:
        self.rows[values[0]] = values

    def update(self, key: int, values: Dict[int, Any]) -> None:
        if key in self.rows:
            row = list(self.rows[key])
            for index, value in values.items():
                row[index] = value
            self.rows[key] = tuple(row)

    def delete(self, key: int) -> None:
        self.rows.pop(key, None)

    def select(
        self,
        where: Sequence[Tuple[int, str, Any]],
        order,
        limit: Optional[int],
        offset: int = 0,
    ) -> List[Tuple[Any, ...]]:
        rows = [row for row in self.rows.values() if self._matches(row, where)]
        if order is None:
            rows.sort(key=lambda row: row[0])
        elif order[0] == "distance":
            rows.sort(key=lambda row: (self._distance(row, order[1]), row[0]))
        else:
            index, descending = order
            if descending:
                # the ties keep the primary key in ascending order
                rows.sort(key=lambda row: row[0])
                rows.sort(key=lambda row: row[index], reverse=True)
            else:
                rows.sort(key=lambda row: (row[index], row[0]))
        if offset:
            rows = rows[offset:]
        if limit is not None:
            rows = rows[:limit]
        return rows

    @staticmethod
    def _matches(row, where) -> bool:
        for column, op, literal in where:
            value = row[column]
            if op == "=":
                if value != literal:
                    return False
            elif op == "<":
                if not value < literal:
                    return False
            elif op == "<=":
                if not value <= literal:
                    return False
            elif op == ">":
                if not value > literal:
                    return False
            else:
                if not value >= literal:
                    return False
        return True

    @staticmethod
    def _distance(row, query) -> float:
        total = 0.0
        for value, other in zip(row[3], query):
            difference = float32(value) - float32(other)
            total += difference * difference
        return math.sqrt(total)


class RandomQueryTest(unittest.TestCase):
    """The engine and the naive reference answer the same."""

    def setUp(self) -> None:
        self.data_dir = Path(tempfile.mkdtemp(prefix="minivectordb-fuzz-"))
        self.database = Database(self.data_dir)
        self.database.open()
        self.engine = Executor(self.database)
        self.engine.execute(
            parse(
                "CREATE TABLE t (id BIGINT PRIMARY KEY, g BIGINT, s VARCHAR(16), "
                "v VECTOR({}));".format(DIM)
            )
        )
        self.reference = Reference()

    def tearDown(self) -> None:
        try:
            self.database.close()
        finally:
            shutil.rmtree(self.data_dir, ignore_errors=True)

    def _run(self, sql):
        return self.engine.execute(parse(sql))

    def _compare(self, name, actual, expected) -> None:
        self.assertEqual(len(actual), len(expected), name)
        for row, want in zip(actual, expected):
            self._compare_value(name, row, want)

    def _compare_value(self, name, value, other) -> None:
        if isinstance(value, list) or isinstance(other, list):
            # a projected vector is a list of numbers
            self.assertEqual(len(value), len(other), name)
            for item, expected_item in zip(value, other):
                self._compare_value(name, item, expected_item)
        elif isinstance(other, float) or isinstance(value, float):
            self.assertLessEqual(
                abs(float(value) - float(other)),
                TOLERANCE * max(1.0, abs(float(other))),
                name,
            )
        else:
            self.assertEqual(value, other, name)

    def test_random_changes_and_queries(self) -> None:
        rng = random.Random(20261112)
        for _ in range(120):
            self._round(rng)

    def _round(self, rng: random.Random) -> None:
        kind = rng.choice(["insert", "insert", "insert", "update", "delete"])
        in_transaction = False
        snapshot = None
        if rng.random() < 0.25 and self.reference.rows:
            snapshot = dict(self.reference.rows)
            self._run("BEGIN;")
            in_transaction = True
        if kind == "insert":
            key = rng.randrange(0, 40)
            group = rng.randrange(0, 3)
            text = rng.choice(TEXTS)
            vector = [float32(rng.uniform(-2, 2)) for _ in range(DIM)]
            try:
                self._run(
                    "INSERT INTO t VALUES ({},{},'{}','{}');".format(
                        key, group, text, "[" + ",".join(str(v) for v in vector) + "]"
                    )
                )
            except Exception as error:  # noqa: BLE001
                self.assertIn("duplicate", str(error))
                self.assertIn(key, self.reference.rows)
                if in_transaction:
                    self._run("ROLLBACK;")
                    self.reference.rows = dict(snapshot)
                return
            self.reference.insert((key, group, text, vector))
        elif kind == "update":
            if not self.reference.rows:
                return
            key = rng.choice(sorted(self.reference.rows))
            group = rng.randrange(0, 3)
            text = rng.choice(TEXTS)
            self._run("UPDATE t SET g = {}, s = '{}' WHERE id = {};".format(group, text, key))
            self.reference.update(key, {1: group, 2: text})
        else:
            if not self.reference.rows:
                return
            key = rng.choice(sorted(self.reference.rows))
            self._run("DELETE FROM t WHERE id = {};".format(key))
            self.reference.delete(key)

        where = []
        for _ in range(rng.randrange(0, 3)):
            if rng.random() < 0.5:
                where.append((1, rng.choice(["=", "<", "<=", ">", ">="]), rng.randrange(0, 3)))
            else:
                where.append(
                    (2, rng.choice(["=", "<", "<="]), rng.choice(TEXTS))
                )
        condition = ""
        if where:
            parts = []
            for column, op, literal in where:
                if column == 1:
                    parts.append("g {} {}".format(op, literal))
                else:
                    parts.append("s {} '{}'".format(op, literal))
            condition = " WHERE " + " AND ".join(parts)

        order = None
        order_text = ""
        if rng.random() < 0.7:
            if rng.random() < 0.5:
                query_vector = [float32(rng.uniform(-2, 2)) for _ in range(DIM)]
                order = ("distance", query_vector)
                order_text = " ORDER BY L2_DISTANCE(v,'[{}]')".format(
                    ",".join(str(v) for v in query_vector)
                )
            else:
                descending = rng.random() < 0.5
                order = (1, descending)
                order_text = " ORDER BY g" + (" DESC" if descending else "")

        limit = rng.choice([None, 1, 2, 5])
        offset = rng.choice([0, 0, 0, 1, 2])
        limit_text = ""
        effective_limit = limit
        effective_offset = offset
        if order is None:
            # A query without an ORDER BY has no defined row order, so a limit
            # would pick an arbitrary subset of the rows.  The public cases
            # always put an ORDER BY in front of a limit.
            effective_limit = None
            effective_offset = 0
        elif limit is not None:
            limit_text = " LIMIT {}".format(limit)
            if offset:
                limit_text = " LIMIT {}, {}".format(offset, limit)
                effective_offset = offset
        else:
            effective_offset = 0

        statement = "SELECT id, g, s FROM t{}{}{};".format(condition, order_text, limit_text)
        expected = self.reference.select(where, order, effective_limit, effective_offset)
        actual = self._run(statement)["rows"]
        want = [[row[0], row[1], row[2]] for row in expected]
        if order is None:
            self._compare(
                statement,
                sorted(actual, key=lambda row: row[0]),
                sorted(want, key=lambda row: row[0]),
            )
        else:
            self._compare(statement, actual, want)

        # The same query, but the vector is projected as well.
        vector_statement = "SELECT id, v FROM t{}{}{};".format(
            condition, order_text, limit_text
        )
        vector_actual = self._run(vector_statement)["rows"]
        vector_want = [[row[0], list(row[3])] for row in expected]
        if order is None:
            self._compare(
                vector_statement,
                sorted(vector_actual, key=lambda row: row[0]),
                sorted(vector_want, key=lambda row: row[0]),
            )
        else:
            self._compare(vector_statement, vector_actual, vector_want)

        if in_transaction:
            inside = self.reference.select([], None, None)
            self._compare(
                "inside the transaction",
                self._run("SELECT id FROM t ORDER BY id;")["rows"],
                [[row[0]] for row in inside],
            )
            self._run("ROLLBACK;")
            self.reference.rows = dict(snapshot)
            after = self.reference.select([], None, None)
            self._compare(
                "after the rollback",
                self._run("SELECT id FROM t ORDER BY id;")["rows"],
                [[row[0]] for row in after],
            )


if __name__ == "__main__":
    unittest.main()
