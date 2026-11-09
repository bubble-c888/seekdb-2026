#!/usr/bin/env python3
"""Comparison logic for the MiniVectorDB mysqltest self-test harness.

Pure functions only: no I/O, no subprocesses, no global state.  The module is
importable so that ``tests/test_compare.py`` can unit test it in isolation.

Assertion semantics (see ``mysqltest/README.md``):

* ``{}``                      the statement must succeed (an empty object does
                              NOT mean "ignore failures").
* ``affected_rows``           integer, must match exactly.
* ``columns``                 list of column names, order matters, aliases
                              included.
* ``rows``                    list of rows compared element-wise in order.
* ``unordered: true``         rows are compared as a multiset: order is
                              ignored but duplicate multiplicity is kept.
* Fields present in the expectation are compared, absent fields are not
  constrained.
* BIGINT values are integers, strings compare by Unicode code point.
* Floats (the ``distance`` column) use BOTH an absolute and a relative
  tolerance of 1e-5; integers, column names and strings must match exactly.
"""

from __future__ import annotations

import json
from typing import Any, List, Sequence

__all__ = [
    "TOLERANCE",
    "values_equal",
    "compare_rows",
    "compare_result",
    "format_value",
]

#: Absolute *and* relative tolerance used for floating point comparisons.
TOLERANCE = 1e-5


def _is_number(value: Any) -> bool:
    """True for int/float values, excluding ``bool``."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_integer(value: Any) -> bool:
    """True for real integers, excluding ``bool``."""
    return isinstance(value, int) and not isinstance(value, bool)


def format_value(value: Any) -> str:
    """Render a value for a human readable mismatch message."""
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=False)
    except (TypeError, ValueError):
        return repr(value)


def values_equal(expected: Any, actual: Any) -> bool:
    """Compare two decoded JSON scalars/containers.

    * two numbers (int/float, never ``bool``) match when
      ``abs(actual - expected) <= 1e-5 * max(1.0, abs(expected))`` which is an
      absolute *and* relative tolerance of 1e-5;
    * ``True``/``False`` never equal ``1``/``0``;
    * lists/tuples are compared element-wise, dicts by key;
    * everything else compares by exact equality.
    """
    if isinstance(expected, bool) or isinstance(actual, bool):
        return (
            isinstance(expected, bool)
            and isinstance(actual, bool)
            and expected == actual
        )

    if _is_number(expected) and _is_number(actual):
        tolerance = TOLERANCE * max(1.0, abs(float(expected)))
        return abs(float(actual) - float(expected)) <= tolerance

    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        if len(expected) != len(actual):
            return False
        return all(
            values_equal(left, right) for left, right in zip(expected, actual)
        )

    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected.keys()) != set(actual.keys()):
            return False
        return all(values_equal(expected[key], actual[key]) for key in expected)

    try:
        return bool(expected == actual)
    except Exception:  # pragma: no cover - defensive, exotic objects only
        return False


def _row_equal(left: Any, right: Any) -> bool:
    """Compare two rows (lists of values) element-wise."""
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return False
        return all(values_equal(a, b) for a, b in zip(left, right))
    return values_equal(left, right)


def _describe_cell_mismatches(expected_row: Any, actual_row: Any) -> List[str]:
    """Describe the individual cells that differ between two rows."""
    if not (
        isinstance(expected_row, (list, tuple))
        and isinstance(actual_row, (list, tuple))
    ):
        return []
    problems = []
    for column, (expected, actual) in enumerate(zip(expected_row, actual_row)):
        if not values_equal(expected, actual):
            problems.append(
                "column {} mismatch: expected {}, got {}".format(
                    column, format_value(expected), format_value(actual)
                )
            )
    return problems


def _match_rows_unordered(
    expected_rows: Sequence[Any], actual_rows: Sequence[Any]
):
    """Maximum bipartite matching between expected and actual rows.

    Returns ``(matched_expected, matched_actual)`` as sets of indices.  Using a
    maximum matching (instead of a greedy scan) keeps duplicate multiplicity
    correct for multiset comparison.
    """
    # match_actual[actual_index] = expected_index
    match_actual: List[int] = [-1] * len(actual_rows)

    def assign(expected_index: int, visited: List[bool]) -> bool:
        for actual_index in range(len(actual_rows)):
            if visited[actual_index]:
                continue
            if not _row_equal(expected_rows[expected_index], actual_rows[actual_index]):
                continue
            visited[actual_index] = True
            current = match_actual[actual_index]
            if current == -1 or assign(current, visited):
                match_actual[actual_index] = expected_index
                return True
        return False

    for expected_index in range(len(expected_rows)):
        assign(expected_index, [False] * len(actual_rows))

    matched_expected = set()
    matched_actual = set()
    for actual_index, expected_index in enumerate(match_actual):
        if expected_index != -1:
            matched_expected.add(expected_index)
            matched_actual.add(actual_index)
    return matched_expected, matched_actual


def compare_rows(
    expected_rows: Sequence[Any],
    actual_rows: Sequence[Any],
    unordered: bool = False,
) -> List[str]:
    """Compare two lists of rows, returning a list of mismatch descriptions.

    An empty list means the rows match.
    """
    if not isinstance(expected_rows, (list, tuple)):
        return ["expected 'rows' to be a list, got {}".format(format_value(expected_rows))]
    if not isinstance(actual_rows, (list, tuple)):
        return ["expected the response 'rows' field to be a list, got {}".format(format_value(actual_rows))]

    problems: List[str] = []
    if unordered:
        matched_expected, matched_actual = _match_rows_unordered(
            expected_rows, actual_rows
        )
        for index, row in enumerate(expected_rows):
            if index not in matched_expected:
                problems.append(
                    "missing row (unordered): {}".format(format_value(row))
                )
        for index, row in enumerate(actual_rows):
            if index not in matched_actual:
                problems.append(
                    "extra row (unordered): {}".format(format_value(row))
                )
        return problems

    if len(expected_rows) != len(actual_rows):
        problems.append(
            "row count mismatch: expected {} rows, got {}".format(
                len(expected_rows), len(actual_rows)
            )
        )
    for index in range(min(len(expected_rows), len(actual_rows))):
        expected_row = expected_rows[index]
        actual_row = actual_rows[index]
        if _row_equal(expected_row, actual_row):
            continue
        cell_problems = _describe_cell_mismatches(expected_row, actual_row)
        detail = "; ".join(cell_problems) if cell_problems else "row differs"
        problems.append(
            "row {} mismatch ({}): expected {}, got {}".format(
                index, detail, format_value(expected_row), format_value(actual_row)
            )
        )
    return problems


def _compare_affected_rows(expected: Any, actual: Any) -> List[str]:
    """``affected_rows`` is an integer and must match exactly."""
    if not _is_integer(expected):
        return [
            "malformed expectation: 'affected_rows' must be an integer, got {}".format(
                format_value(expected)
            )
        ]
    if not _is_integer(actual):
        return [
            "affected_rows mismatch: expected {}, got {}".format(
                format_value(expected), format_value(actual)
            )
        ]
    if expected != actual:
        return [
            "affected_rows mismatch: expected {}, got {}".format(
                format_value(expected), format_value(actual)
            )
        ]
    return []


def _compare_columns(expected: Any, actual: Any) -> List[str]:
    """Column names are compared exactly and in order."""
    if not isinstance(expected, (list, tuple)):
        return [
            "malformed expectation: 'columns' must be a list, got {}".format(
                format_value(expected)
            )
        ]
    if not isinstance(actual, (list, tuple)):
        return [
            "columns mismatch: expected {}, but the response has no 'columns' field".format(
                format_value(list(expected))
            )
        ]
    if len(expected) != len(actual):
        return [
            "columns mismatch: expected {} column(s) {}, got {} {}".format(
                len(expected),
                format_value(list(expected)),
                len(actual),
                format_value(list(actual)),
            )
        ]
    problems = []
    for index, (expected_name, actual_name) in enumerate(zip(expected, actual)):
        if expected_name != actual_name:
            problems.append(
                "column name mismatch at position {}: expected {}, got {}".format(
                    index, format_value(expected_name), format_value(actual_name)
                )
            )
    return problems


def compare_result(expected: dict, actual: dict) -> List[str]:
    """Compare one expected assertion object with one actual response object.

    Returns a list of human readable mismatch descriptions; an empty list means
    the response matches the expectation.
    """
    if not isinstance(expected, dict):
        return [
            "malformed expectation: expected a JSON object, got {}".format(
                format_value(expected)
            )
        ]
    if not isinstance(actual, dict):
        return [
            "malformed response: expected a JSON object, got {}".format(
                format_value(actual)
            )
        ]

    problems: List[str] = []

    if "error" in actual:
        message = actual.get("error")
        if "error" in expected:
            if expected.get("error") != message:
                problems.append(
                    "error message mismatch: expected {}, got {}".format(
                        format_value(expected.get("error")), format_value(message)
                    )
                )
            for key in ("code", "sqlstate"):
                if key in expected and expected.get(key) != actual.get(key):
                    problems.append(
                        "{} mismatch: expected {}, got {}".format(
                            key,
                            format_value(expected.get(key)),
                            format_value(actual.get(key)),
                        )
                    )
            return problems
        details = "error: {}".format(message)
        if actual.get("code") is not None:
            details += ", code: {}".format(format_value(actual.get("code")))
        if actual.get("sqlstate") is not None:
            details += ", sqlstate: {}".format(format_value(actual.get("sqlstate")))
        return ["statement failed but should have succeeded ({})".format(details)]

    if "affected_rows" in expected:
        problems.extend(
            _compare_affected_rows(expected["affected_rows"], actual.get("affected_rows"))
        )

    if "columns" in expected:
        problems.extend(_compare_columns(expected["columns"], actual.get("columns")))

    if "rows" in expected:
        unordered = bool(expected.get("unordered", False))
        actual_rows = actual.get("rows")
        if actual_rows is None:
            problems.append(
                "expected rows {} but the response has no 'rows' field".format(
                    format_value(expected["rows"])
                )
            )
        else:
            problems.extend(
                compare_rows(expected["rows"], actual_rows, unordered=unordered)
            )

    return problems
