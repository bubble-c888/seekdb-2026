#!/usr/bin/env python3
"""Unit tests for the MiniVectorDB result comparison logic (tools/compare.py)."""

import os
import sys
import unittest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TOOLS = os.path.join(_REPO_ROOT, "tools")
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

from compare import compare_result, compare_rows, values_equal  # noqa: E402


class ValuesEqualTest(unittest.TestCase):
    def test_exact_integers(self):
        self.assertTrue(values_equal(3, 3))
        self.assertFalse(values_equal(3, 4))

    def test_absolute_tolerance(self):
        self.assertTrue(values_equal(1.0, 1.000009))
        self.assertFalse(values_equal(1.0, 1.0001))

    def test_relative_tolerance(self):
        self.assertTrue(values_equal(1000000.0, 1000005.0))
        self.assertFalse(values_equal(1000000.0, 1000100.0))

    def test_zero_uses_absolute_tolerance(self):
        self.assertTrue(values_equal(0.0, 0.0))
        self.assertFalse(values_equal(0.0, 0.001))

    def test_int_and_float_are_both_numbers(self):
        self.assertTrue(values_equal(1, 1.0))
        self.assertFalse(values_equal(1, 1.5))

    def test_bool_is_not_a_number(self):
        self.assertFalse(values_equal(True, 1))
        self.assertFalse(values_equal(False, 0))
        self.assertFalse(values_equal(True, 1.0))
        self.assertTrue(values_equal(True, True))
        self.assertFalse(values_equal(True, False))

    def test_strings_compare_exactly(self):
        self.assertTrue(values_equal("甲", "甲"))
        self.assertFalse(values_equal("甲", "乙"))
        self.assertFalse(values_equal("a", "A"))

    def test_none_and_other_types(self):
        self.assertTrue(values_equal(None, None))
        self.assertFalse(values_equal(None, 0))
        self.assertFalse(values_equal("1", 1))

    def test_lists_compare_element_wise(self):
        self.assertTrue(values_equal([1.0, 2.0], [1.0, 2.000001]))
        self.assertFalse(values_equal([1.0, 2.0], [1.0, 2.0, 3.0]))


class CompareRowsTest(unittest.TestCase):
    def test_ordered_match(self):
        self.assertEqual(compare_rows([[1], [2]], [[1], [2]], False), [])

    def test_ordered_detects_wrong_order(self):
        problems = compare_rows([[1], [2]], [[2], [1]], False)
        self.assertTrue(problems)
        self.assertIn("row 0 mismatch", problems[0])

    def test_ordered_detects_length_mismatch(self):
        problems = compare_rows([[1]], [[1], [2]], False)
        self.assertTrue(problems)
        self.assertIn("row count mismatch: expected 1 rows, got 2", problems)

    def test_ordered_reports_cell_mismatch(self):
        problems = compare_rows([[1, "a"]], [[1, "b"]], False)
        self.assertTrue(problems)
        self.assertIn("column 1 mismatch", problems[0])

    def test_empty_rows(self):
        self.assertEqual(compare_rows([], [], False), [])
        self.assertEqual(compare_rows([], [], True), [])

    def test_unordered_ignores_order(self):
        self.assertEqual(compare_rows([[1], [2]], [[2], [1]], True), [])

    def test_unordered_keeps_duplicate_multiplicity(self):
        problems = compare_rows([[1], [1], [2]], [[1], [2]], True)
        self.assertEqual(len(problems), 1)
        self.assertIn("missing row (unordered): [1]", problems[0])

    def test_unordered_reports_extra_rows(self):
        problems = compare_rows([[1]], [[1], [2], [2]], True)
        self.assertEqual(len(problems), 2)
        self.assertTrue(all("extra row (unordered): [2]" in p for p in problems))

    def test_unordered_reports_missing_and_extra(self):
        problems = compare_rows([[1], [3]], [[1], [4]], True)
        self.assertEqual(len(problems), 2)
        self.assertTrue(any("missing row (unordered): [3]" in p for p in problems))
        self.assertTrue(any("extra row (unordered): [4]" in p for p in problems))

    def test_unordered_float_tolerance(self):
        self.assertEqual(compare_rows([[1, 5.0]], [[1, 5.000001]], True), [])

    def test_rows_must_be_lists(self):
        self.assertTrue(compare_rows("nope", [], False))


class CompareResultTest(unittest.TestCase):
    def test_empty_expectation_requires_success(self):
        self.assertEqual(compare_result({}, {}), [])
        self.assertEqual(compare_result({}, {"affected_rows": 0}), [])

    def test_empty_expectation_rejects_error(self):
        problems = compare_result({}, {"error": "boom", "code": 1064, "sqlstate": "42000"})
        self.assertEqual(len(problems), 1)
        self.assertIn("statement failed but should have succeeded", problems[0])
        self.assertIn("boom", problems[0])

    def test_affected_rows_must_match_exactly(self):
        self.assertEqual(compare_result({"affected_rows": 3}, {"affected_rows": 3}), [])
        problems = compare_result({"affected_rows": 3}, {"affected_rows": 2})
        self.assertIn("affected_rows mismatch: expected 3, got 2", problems)

    def test_affected_rows_rejects_float(self):
        problems = compare_result({"affected_rows": 3}, {"affected_rows": 3.0})
        self.assertTrue(problems)
        self.assertIn("affected_rows mismatch", problems[0])

    def test_affected_rows_missing_in_response(self):
        problems = compare_result({"affected_rows": 3}, {})
        self.assertTrue(problems)

    def test_columns_must_match_in_order(self):
        expected = {"columns": ["id", "content"]}
        self.assertEqual(compare_result(expected, {"columns": ["id", "content"]}), [])
        problems = compare_result(expected, {"columns": ["content", "id"]})
        self.assertIn("column name mismatch at position 0: expected \"id\", got \"content\"", problems)

    def test_columns_count_mismatch(self):
        problems = compare_result({"columns": ["id", "content"]}, {"columns": ["id"]})
        self.assertTrue(problems)
        self.assertIn("columns mismatch", problems[0])

    def test_rows_are_compared_in_order(self):
        expected = {"columns": ["id"], "rows": [[1], [2]]}
        self.assertEqual(compare_result(expected, {"columns": ["id"], "rows": [[1], [2]]}), [])
        problems = compare_result(expected, {"columns": ["id"], "rows": [[2], [1]]})
        self.assertTrue(problems)

    def test_unordered_flag_uses_multiset(self):
        expected = {"columns": ["category"], "rows": [["甲"], ["乙"], ["甲"]], "unordered": True}
        actual = {"columns": ["category"], "rows": [["甲"], ["甲"], ["乙"]]}
        self.assertEqual(compare_result(expected, actual), [])

    def test_unordered_flag_still_checks_duplicates(self):
        expected = {"columns": ["category"], "rows": [["甲"], ["甲"], ["乙"]], "unordered": True}
        actual = {"columns": ["category"], "rows": [["甲"], ["乙"]]}
        self.assertTrue(compare_result(expected, actual))

    def test_absent_fields_are_not_constrained(self):
        expected = {"columns": ["id"], "rows": [[1]]}
        actual = {"columns": ["id"], "rows": [[1]], "extra": "ignored"}
        self.assertEqual(compare_result(expected, actual), [])

    def test_missing_rows_field_is_reported(self):
        problems = compare_result({"rows": [[1]]}, {"columns": ["id"]})
        self.assertTrue(problems)

    def test_distance_column_uses_tolerance(self):
        expected = {"columns": ["id", "distance"], "rows": [[2, 1.0], [3, 2.0], [1, 5.0]]}
        actual = {
            "columns": ["id", "distance"],
            "rows": [[2, 1.0000001], [3, 1.9999999], [1, 5.0]],
        }
        self.assertEqual(compare_result(expected, actual), [])

    def test_non_object_inputs(self):
        self.assertTrue(compare_result([], {}))
        self.assertTrue(compare_result({}, []))


if __name__ == "__main__":
    unittest.main()
