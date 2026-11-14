"""The partial selection of the best rows has to sort like a complete sort.

The query no longer sorts the whole chunk of rows: it selects the rows that
can enter the running list with a partition and sorts only those.  The rows
that are exactly as far away as the last one that is kept have to be taken
along, because their primary key decides the order between them.

The tests compare the selection with the complete sort of the same chunk, on
data that is full of ties.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from minivectordb.engine.database import Database  # noqa: E402
from minivectordb.engine.executor import Executor  # noqa: E402
from minivectordb.sqlparse import parse  # noqa: E402


class SelectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.executor = Executor.__new__(Executor)

    def test_the_selection_matches_the_complete_sort(self) -> None:
        rng = np.random.default_rng(20261114)
        for _ in range(200):
            size = int(rng.integers(20, 400))
            wanted = int(rng.integers(1, 30))
            # A lot of ties: only a few distinct values.
            values = rng.integers(0, 4, size=size).astype(np.float64)
            pks = rng.permutation(size).astype(np.int64)
            arrays = [values]
            expected = np.lexsort(tuple(reversed(arrays + [pks])))
            actual = self.executor._select_best(arrays, pks, wanted)
            self.assertEqual(
                list(expected[:wanted]),
                list(actual[:wanted]),
                "the selection differs from the complete sort",
            )

    def test_the_selection_matches_with_two_keys(self) -> None:
        rng = np.random.default_rng(20261115)
        for _ in range(200):
            size = int(rng.integers(20, 400))
            wanted = int(rng.integers(1, 30))
            first = rng.integers(0, 3, size=size).astype(np.float64)
            second = rng.integers(0, 3, size=size).astype(np.float64)
            pks = rng.permutation(size).astype(np.int64)
            arrays = [first, second]
            expected = np.lexsort(tuple(reversed(arrays + [pks])))
            actual = self.executor._select_best(arrays, pks, wanted)
            self.assertEqual(list(expected[:wanted]), list(actual[:wanted]))


class TiedOrderTest(unittest.TestCase):
    """The order of the rows that are equally far away."""

    def setUp(self) -> None:
        import shutil
        import tempfile

        self.directory = Path(tempfile.mkdtemp(prefix="minivectordb-ties-"))
        self.database = Database(self.directory)
        self.database.open()
        self.executor = Executor(self.database)

    def tearDown(self) -> None:
        import shutil

        try:
            self.database.close()
        except Exception:
            pass
        shutil.rmtree(self.directory, ignore_errors=True)

    def rows(self, sql: str):
        return self.executor.execute(parse(sql))["rows"]

    def test_equal_distances_are_ordered_by_the_primary_key(self) -> None:
        self.executor.execute(parse("CREATE TABLE t (id BIGINT PRIMARY KEY, e VECTOR(2));"))
        # Four rows at the same distance from the origin, the keys are not in
        # the order of the slots.
        for key in (40, 10, 30, 20):
            self.executor.execute(parse("INSERT INTO t VALUES (%d, '[3.0,4.0]');" % key))
        rows = self.rows(
            "SELECT id FROM t ORDER BY L2_DISTANCE(e,'[0.0,0.0]') LIMIT 4;"
        )
        self.assertEqual([row[0] for row in rows], [10, 20, 30, 40])

    def test_the_offset_does_not_change_the_order(self) -> None:
        self.executor.execute(parse("CREATE TABLE t (id BIGINT PRIMARY KEY, g BIGINT);"))
        for index in range(1, 11):
            self.executor.execute(parse("INSERT INTO t VALUES (%d, %d);" % (index, index % 3)))
        everything = [row[0] for row in self.rows("SELECT id FROM t ORDER BY g, id;")]
        shifted = [row[0] for row in self.rows("SELECT id FROM t ORDER BY g, id LIMIT 2, 5;")]
        self.assertEqual(shifted, everything[2:7])

    def test_a_limit_of_zero_returns_no_rows(self) -> None:
        # The selection used to fail here: the partition of an empty request
        # has nothing to select.
        self.executor.execute(parse("CREATE TABLE t (id BIGINT PRIMARY KEY, g BIGINT);"))
        for index in range(1, 6):
            self.executor.execute(parse("INSERT INTO t VALUES (%d, %d);" % (index, index % 2)))
        self.assertEqual(self.rows("SELECT id FROM t ORDER BY g LIMIT 0;"), [])
        self.assertEqual(self.rows("SELECT id FROM t ORDER BY g, id LIMIT 0;"), [])
        self.assertEqual(self.rows("SELECT id FROM t ORDER BY g LIMIT 0 OFFSET 2;"), [])
        self.assertEqual(self.rows("SELECT id FROM t LIMIT 0;"), [])
        self.assertEqual(self.rows("SELECT id FROM t ORDER BY g LIMIT 1;"), [[2]])

    def test_a_limit_larger_than_the_table_returns_everything(self) -> None:
        self.executor.execute(parse("CREATE TABLE t (id BIGINT PRIMARY KEY, g BIGINT);"))
        for index in range(1, 6):
            self.executor.execute(parse("INSERT INTO t VALUES (%d, %d);" % (index, index % 2)))
        rows = [row[0] for row in self.rows("SELECT id FROM t ORDER BY g, id LIMIT 100;")]
        self.assertEqual(rows, [2, 4, 1, 3, 5])


if __name__ == "__main__":
    unittest.main()
