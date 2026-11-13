"""The expanded distance computation must answer like the previous one.

The distance is computed with the expansion

    ||x - q||^2 = ||x||^2 - 2 x.q + ||q||^2

so that the inner product runs through the matrix product of the library.
The expansion cancels when the distance is small compared to the norms, so
the rows that are close to the query are measured again with the difference
formula.  The tests below pin that behaviour down: the answers have to be
the same, and a row that is equal to the query has to have the distance zero
and not something near it.

Run them with::

    python -m unittest tests.test_distance_expansion -v
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from minivectordb.engine.database import Database  # noqa: E402
from minivectordb.engine.executor import Executor  # noqa: E402
from minivectordb.engine.vector import l2_distances  # noqa: E402
from minivectordb.sqlparse import parse  # noqa: E402


def straightforward(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    """The distance as it is written in the task statement."""
    diff = matrix.astype(np.float64) - query.astype(np.float64)
    return np.sqrt(np.einsum("ij,ij->i", diff, diff))


class DistanceExpansionTest(unittest.TestCase):
    """The expansion against the straightforward computation."""

    def setUp(self) -> None:
        rng = np.random.default_rng(20261113)
        self.rows = 2000
        self.dim = 64
        self.matrix = rng.uniform(-1.0, 1.0, size=(self.rows, self.dim)).astype(np.float32)
        self.query = rng.uniform(-1.0, 1.0, size=self.dim).astype(np.float32)

    def test_the_answers_are_the_same(self) -> None:
        expected = straightforward(self.matrix, self.query)
        actual = l2_distances(self.matrix, self.query)
        self.assertLess(float(np.max(np.abs(expected - actual))), 1e-5)
        # The order of the ten nearest rows must not change.
        self.assertEqual(
            list(np.argsort(expected, kind="stable")[:10]),
            list(np.argsort(actual, kind="stable")[:10]),
        )

    def test_the_order_survives_a_whole_workload(self) -> None:
        rng = np.random.default_rng(20261114)
        for _ in range(200):
            query = rng.uniform(-1.0, 1.0, size=self.dim).astype(np.float32)
            expected = straightforward(self.matrix, query)
            actual = l2_distances(self.matrix, query)
            self.assertEqual(
                list(np.argsort(expected, kind="stable")[:10]),
                list(np.argsort(actual, kind="stable")[:10]),
            )

    def test_a_row_equal_to_the_query_has_the_distance_zero(self) -> None:
        exact = self.matrix[17]
        distance = l2_distances(np.asarray([exact]), exact)[0]
        self.assertEqual(float(distance), 0.0)

    def test_a_row_next_to_an_equal_one_is_still_found(self) -> None:
        exact = self.matrix[17]
        neighbour = exact.copy()
        neighbour[0] = np.float32(exact[0] + 1e-3)
        matrix = np.stack([exact, neighbour])
        distances = l2_distances(matrix, exact)
        self.assertEqual(float(distances[0]), 0.0)
        self.assertAlmostEqual(float(distances[1]), 1e-3, places=6)

    def test_the_distance_of_a_row_to_itself_is_zero_in_sql(self) -> None:
        data_dir = Path(tempfile.mkdtemp(prefix="minivectordb-distance-"))
        database = Database(data_dir)
        database.open()
        executor = Executor(database)
        try:
            executor.execute(parse("CREATE TABLE t (id BIGINT PRIMARY KEY, e VECTOR(4));"))
            vector = "[0.5,-0.25,0.125,1.0]"
            executor.execute(parse("INSERT INTO t VALUES (1, '%s');" % vector))
            answer = executor.execute(
                parse("SELECT L2_DISTANCE(e,'%s') FROM t WHERE id = 1;" % vector)
            )
            self.assertEqual(float(answer["rows"][0][0]), 0.0)
            answer = executor.execute(
                parse("SELECT id FROM t ORDER BY L2_DISTANCE(e,'%s') LIMIT 1;" % vector)
            )
            self.assertEqual(answer["rows"][0][0], 1)
        finally:
            try:
                database.close()
            except Exception:
                pass
            shutil.rmtree(data_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
