"""Unit tests for the MiniVectorDB SQL parser.

Run them with::

    python -m unittest tests.test_parser -v
"""

from __future__ import annotations

import os
import unittest

from minivectordb.errors import ParseError, UnsupportedError
from minivectordb.sqlparse import ast, parse

HERE = os.path.dirname(os.path.abspath(__file__))
CASES_DIR = os.path.normpath(os.path.join(HERE, os.pardir, "mysqltest", "cases"))


def _doc_table(with_vector: bool = False) -> ast.CreateTable:
    columns = [
        ast.ColumnDef("id", "BIGINT", None, True, False),
        ast.ColumnDef("category", "VARCHAR", 8, False, False),
        ast.ColumnDef("content", "VARCHAR", 16, False, False),
    ]
    if with_vector:
        columns.append(ast.ColumnDef("embedding", "VECTOR", 2, False, False))
    return ast.CreateTable(name="docs", columns=columns, primary_key="id")


# ---------------------------------------------------------------------------
# the public sample cases (mysqltest/cases/*.test)
# ---------------------------------------------------------------------------

PUBLIC_CASES_DIR = CASES_DIR
PUBLIC_MANIFEST = os.path.normpath(
    os.path.join(HERE, os.pardir, "mysqltest", "manifest.json")
)


def _public_case_ids():
    """The ids of the public cases, taken from the manifest."""
    import json

    with open(PUBLIC_MANIFEST, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    return [entry["id"] for entry in manifest.get("cases", [])]


def _public_expectations(case_id):
    """The expected result objects of a public case."""
    import json

    path = os.path.join(PUBLIC_CASES_DIR, case_id + ".result")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _public_items(case_id):
    """The items of a public case: ``(kind, text)`` per line."""
    path = os.path.join(PUBLIC_CASES_DIR, case_id + ".test")
    with open(path, "r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]
    items = []
    for line in lines:
        if line.startswith("--"):
            items.append(("directive", line))
        else:
            items.append(("sql", line))
    return items


class PublicCaseTest(unittest.TestCase):
    """Every public sample statement must be understood by the parser."""

    def test_public_cases_exist(self):
        self.assertTrue(os.path.isdir(CASES_DIR))
        ids = _public_case_ids()
        self.assertTrue(ids, "the manifest does not list a single case")
        for name in ids:
            for suffix in (".test", ".result"):
                self.assertTrue(
                    os.path.isfile(os.path.join(CASES_DIR, name + suffix)),
                    name + suffix,
                )

    def test_public_sample_statements(self):
        for name in _public_case_ids():
            statements = [text for kind, text in _public_items(name) if kind == "sql"]
            expected = _public_expectations(name)
            self.assertEqual(len(statements), len(expected), name)
            for index, sql in enumerate(statements):
                with self.subTest(case=name, index=index, sql=sql):
                    try:
                        parse(sql)
                    except UnsupportedError:
                        # A statement outside of the supported subset is
                        # rejected by the executor, not by the parser.
                        self.assertEqual(
                            expected[index].get("error"), "unsupported_sql"
                        )

    def test_public_expectations_match_the_manifest(self):
        import json

        with open(PUBLIC_MANIFEST, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        for entry in manifest.get("cases", []):
            name = entry["id"]
            statements = [text for kind, text in _public_items(name) if kind == "sql"]
            self.assertEqual(
                len(statements), int(entry.get("statements", 0)), name
            )

    def test_first_statement_of_sql_crud(self):
        statement = parse(
            "CREATE TABLE docs (id BIGINT PRIMARY KEY, content VARCHAR(16));"
        )
        self.assertEqual(
            statement,
            ast.CreateTable(
                name="docs",
                columns=[
                    ast.ColumnDef("id", "BIGINT", None, True, False),
                    ast.ColumnDef("content", "VARCHAR", 16, False, False),
                ],
                primary_key="id",
            ),
        )


class GrammarTest(unittest.TestCase):
    """The statements of the task description."""

    def test_create_table(self):
        stmt = parse(
            "CREATE TABLE docs (id BIGINT PRIMARY KEY, category VARCHAR(8), "
            "content VARCHAR(16));"
        )
        self.assertEqual(
            stmt,
            ast.CreateTable(
                name="docs",
                columns=[
                    ast.ColumnDef("id", "BIGINT", None, True, False),
                    ast.ColumnDef("category", "VARCHAR", 8, False, False),
                    ast.ColumnDef("content", "VARCHAR", 16, False, False),
                ],
                primary_key="id",
            ),
        )

    def test_create_table_with_vector(self):
        stmt = parse(
            "CREATE TABLE docs (id BIGINT PRIMARY KEY, embedding VECTOR(2));"
        )
        self.assertEqual(
            stmt,
            ast.CreateTable(
                name="docs",
                columns=[
                    ast.ColumnDef("id", "BIGINT", None, True, False),
                    ast.ColumnDef("embedding", "VECTOR", 2, False, False),
                ],
                primary_key="id",
            ),
        )

    def test_create_table_not_null(self):
        stmt = parse(
            "CREATE TABLE docs (id BIGINT PRIMARY KEY NOT NULL, "
            "note VARCHAR(4) NOT NULL);"
        )
        self.assertEqual(
            stmt,
            ast.CreateTable(
                name="docs",
                columns=[
                    ast.ColumnDef("id", "BIGINT", None, True, True),
                    ast.ColumnDef("note", "VARCHAR", 4, False, True),
                ],
                primary_key="id",
            ),
        )

    def test_create_table_primary_key_before_not_null(self):
        stmt = parse("CREATE TABLE docs (id BIGINT PRIMARY KEY NOT NULL);")
        self.assertTrue(stmt.columns[0].primary_key)
        self.assertTrue(stmt.columns[0].not_null)

    def test_create_table_table_level_primary_key(self):
        stmt = parse(
            "CREATE TABLE docs (id BIGINT, note VARCHAR(4), PRIMARY KEY (id));"
        )
        self.assertEqual(
            stmt,
            ast.CreateTable(
                name="docs",
                columns=[
                    ast.ColumnDef("id", "BIGINT", None, True, False),
                    ast.ColumnDef("note", "VARCHAR", 4, False, False),
                ],
                primary_key="id",
            ),
        )

    def test_insert_multi_row(self):
        stmt = parse("INSERT INTO docs VALUES (3, 'a'), (1, 'b');")
        self.assertEqual(
            stmt,
            ast.Insert(
                table="docs",
                columns=None,
                rows=[
                    [ast.IntLiteral(3), ast.StrLiteral("a")],
                    [ast.IntLiteral(1), ast.StrLiteral("b")],
                ],
            ),
        )

    def test_insert_with_columns(self):
        stmt = parse("INSERT INTO docs (id, note) VALUES (1, 'a');")
        self.assertEqual(
            stmt,
            ast.Insert(
                table="docs",
                columns=["id", "note"],
                rows=[[ast.IntLiteral(1), ast.StrLiteral("a")]],
            ),
        )

    def test_select_star_order_by(self):
        stmt = parse("SELECT * FROM docs ORDER BY id;")
        self.assertEqual(
            stmt,
            ast.Select(
                table="docs",
                items=[ast.SelectItem(expr=ast.ColumnRef("*"))],
                where=None,
                order_by=[ast.OrderItem("id", False)],
                limit=None,
            ),
        )

    def test_select_with_distance_where_order_limit(self):
        stmt = parse(
            "SELECT id, L2_DISTANCE(embedding, '[0, 0]') AS distance FROM docs "
            "WHERE id > 1 AND id <= 3 ORDER BY distance LIMIT 10;"
        )
        self.assertEqual(
            stmt,
            ast.Select(
                table="docs",
                items=[
                    ast.SelectItem(expr=ast.ColumnRef("id")),
                    ast.SelectItem(
                        expr=ast.DistanceFunc(column="embedding", vector="[0, 0]"),
                        alias="distance",
                    ),
                ],
                where=ast.And(
                    ast.Compare(ast.ColumnRef("id"), ">", ast.IntLiteral(1)),
                    ast.Compare(ast.ColumnRef("id"), "<=", ast.IntLiteral(3)),
                ),
                order_by=[ast.OrderItem("distance", False)],
                limit=10,
            ),
        )

    def test_update(self):
        stmt = parse("UPDATE docs SET content = 'x' WHERE category = 'a';")
        self.assertEqual(
            stmt,
            ast.Update(
                table="docs",
                assignments=[ast.Assignment("content", ast.StrLiteral("x"))],
                where=ast.Compare(
                    ast.ColumnRef("category"), "=", ast.StrLiteral("a")
                ),
            ),
        )

    def test_update_multiple_assignments(self):
        stmt = parse("UPDATE docs SET a = 1, b = 'x' WHERE id = 2;")
        self.assertEqual(
            stmt,
            ast.Update(
                table="docs",
                assignments=[
                    ast.Assignment("a", ast.IntLiteral(1)),
                    ast.Assignment("b", ast.StrLiteral("x")),
                ],
                where=ast.Compare(ast.ColumnRef("id"), "=", ast.IntLiteral(2)),
            ),
        )

    def test_delete(self):
        stmt = parse("DELETE FROM docs WHERE id >= 2;")
        self.assertEqual(
            stmt,
            ast.Delete(
                table="docs",
                where=ast.Compare(ast.ColumnRef("id"), ">=", ast.IntLiteral(2)),
            ),
        )

    def test_transactions(self):
        self.assertEqual(parse("BEGIN;"), ast.Begin())
        self.assertEqual(parse("COMMIT;"), ast.Commit())
        self.assertEqual(parse("ROLLBACK;"), ast.Rollback())
        self.assertEqual(parse("begin"), ast.Begin())
        self.assertEqual(parse("commit"), ast.Commit())
        self.assertEqual(parse("rollback"), ast.Rollback())

    def test_create_vector_index(self):
        stmt = parse("CREATE VECTOR INDEX idx_docs ON docs(embedding);")
        self.assertEqual(
            stmt,
            ast.CreateVectorIndex(name="idx_docs", table="docs", column="embedding"),
        )

    def test_limit_zero(self):
        stmt = parse("SELECT id FROM docs LIMIT 0;")
        self.assertEqual(stmt.limit, 0)

    def test_and_is_left_associative(self):
        stmt = parse("SELECT id FROM docs WHERE a = 1 AND b = 2 AND c = 3;")
        self.assertIsInstance(stmt.where, ast.And)
        self.assertIsInstance(stmt.where.left, ast.And)
        self.assertIsInstance(stmt.where.right, ast.Compare)


class LexerFeatureTest(unittest.TestCase):
    """Comments, whitespace, case folding, escapes and quoting."""

    def test_line_comments(self):
        stmt = parse(
            "-- a leading comment\n"
            "SELECT id FROM docs; # a trailing comment\n"
        )
        self.assertEqual(stmt.table, "docs")
        self.assertEqual(len(stmt.items), 1)

    def test_block_comment(self):
        stmt = parse("SELECT /* the id */ id FROM /* docs */ docs;")
        self.assertEqual(stmt.table, "docs")
        self.assertEqual(stmt.items[0].expr.name, "id")

    def test_extra_whitespace_and_newlines(self):
        stmt = parse("  SELECT\n\t id \r\n FROM   docs  ;  ")
        self.assertEqual(stmt.table, "docs")

    def test_mixed_case_keywords(self):
        stmt = parse("sElEcT iD fRoM dOcS wHeRe iD > 1 oRdEr By iD lImIt 2")
        self.assertEqual(stmt.table, "dOcS")
        self.assertEqual(stmt.items[0].expr.name, "iD")
        self.assertEqual(stmt.limit, 2)

    def test_identifier_case_is_preserved(self):
        stmt = parse("SELECT ID FROM Docs")
        self.assertEqual(stmt.table, "Docs")
        self.assertEqual(stmt.items[0].expr.name, "ID")

    def test_escaped_quote(self):
        stmt = parse("INSERT INTO docs (note) VALUES ('it''s');")
        self.assertEqual(stmt.rows[0][0], ast.StrLiteral("it's"))

    def test_backslash_escapes(self):
        stmt = parse(r"INSERT INTO docs (note) VALUES ('a\'b\\c\nd\te\rf\0g');")
        self.assertEqual(
            stmt.rows[0][0], ast.StrLiteral("a'b\\c\nd\te\rf\0g")
        )

    def test_unknown_backslash_escape_keeps_backslash(self):
        stmt = parse(r"INSERT INTO docs (note) VALUES ('a\qb');")
        self.assertEqual(stmt.rows[0][0], ast.StrLiteral("a\\qb"))

    def test_negative_number(self):
        stmt = parse("UPDATE docs SET score = -7 WHERE id = -1;")
        self.assertEqual(stmt.assignments[0].value, ast.IntLiteral(-7))
        self.assertEqual(stmt.where.right, ast.IntLiteral(-1))

    def test_positive_number(self):
        stmt = parse("UPDATE docs SET score = +7 WHERE id = 1;")
        self.assertEqual(stmt.assignments[0].value, ast.IntLiteral(7))

    def test_backquoted_identifiers(self):
        stmt = parse("SELECT `id` FROM `docs`")
        self.assertEqual(stmt.table, "docs")
        self.assertEqual(stmt.items[0].expr.name, "id")

    def test_trailing_semicolon_is_optional(self):
        self.assertEqual(parse("BEGIN"), parse("BEGIN;"))


class ErrorTest(unittest.TestCase):
    """Errors and unsupported constructs."""

    def assertParseError(self, sql, incomplete=None):
        with self.assertRaises(ParseError) as ctx:
            parse(sql)
        self.assertTrue(hasattr(ctx.exception, "incomplete"))
        if incomplete is not None:
            self.assertEqual(ctx.exception.incomplete, incomplete)
        return ctx.exception

    def assertUnsupported(self, sql):
        with self.assertRaises(UnsupportedError) as ctx:
            parse(sql)
        self.assertTrue(hasattr(ctx.exception, "incomplete"))
        self.assertFalse(ctx.exception.incomplete)
        return ctx.exception

    def test_unsupported_leading_keywords(self):
        for sql in [
            "ALTER TABLE docs ADD COLUMN x BIGINT",
            "DROP TABLE docs",
            "TRUNCATE TABLE docs",
            "RENAME TABLE docs TO other",
            "SET autocommit = 1",
            "SHOW TABLES",
            "DESCRIBE docs",
            "EXPLAIN SELECT id FROM docs",
            "USE otherdb",
            "GRANT ALL ON docs TO bob",
            "REVOKE ALL ON docs FROM bob",
            "WITH x AS (SELECT 1) SELECT * FROM x",
            "SELECT id FROM docs UNION SELECT id FROM other",
            "SELECT id FROM docs INTERSECT SELECT id FROM other",
            "SELECT id FROM docs EXCEPT SELECT id FROM other",
            "REPLACE INTO docs VALUES (1)",
            "CALL my_proc()",
            "PREPARE stmt FROM 'SELECT 1'",
            "EXECUTE stmt",
            "SAVEPOINT sp1",
            "LOCK TABLES docs READ",
            "ANALYZE TABLE docs",
            "OPTIMIZE TABLE docs",
            "CREATE INDEX idx ON docs(id)",
            "CREATE DATABASE db",
            "CREATE SCHEMA db",
            "CREATE VIEW v AS SELECT id FROM docs",
        ]:
            with self.subTest(sql=sql):
                self.assertUnsupported(sql)

    def test_unsupported_code(self):
        err = self.assertUnsupported("ALTER TABLE docs ADD COLUMN x BIGINT")
        self.assertEqual(err.code, 1235)

    def test_parse_error_code(self):
        err = self.assertParseError("SELECT a, * FROM t")
        self.assertEqual(err.code, 1064)

    def test_or_in_where(self):
        self.assertUnsupported("SELECT id FROM docs WHERE a = 1 OR b = 2")

    def test_aggregate_function(self):
        self.assertUnsupported("SELECT COUNT(*) FROM docs")
        self.assertUnsupported("SELECT SUM(id) FROM docs")
        self.assertUnsupported("SELECT AVG(id) FROM docs")
        self.assertUnsupported("SELECT MIN(id) FROM docs")
        self.assertUnsupported("SELECT MAX(id) FROM docs")

    def test_star_with_other_item_is_a_syntax_error(self):
        self.assertParseError("SELECT a, * FROM t")

    def test_select_without_from(self):
        self.assertUnsupported("SELECT id")
        self.assertUnsupported("SELECT *")

    def test_in_between_like_is_null(self):
        self.assertUnsupported("SELECT id FROM docs WHERE id IN (1, 2)")
        self.assertUnsupported("SELECT id FROM docs WHERE id BETWEEN 1 AND 2")
        self.assertUnsupported("SELECT id FROM docs WHERE note LIKE 'a%'")
        self.assertUnsupported("SELECT id FROM docs WHERE note IS NULL")
        self.assertUnsupported("SELECT id FROM docs WHERE note IS NOT NULL")
        self.assertUnsupported("SELECT id FROM docs WHERE NOT id = 1")
        self.assertUnsupported("SELECT id FROM docs WHERE note = NULL")

    def test_group_by_having_distinct(self):
        self.assertUnsupported("SELECT id FROM docs GROUP BY id")
        self.assertUnsupported("SELECT id FROM docs GROUP BY id HAVING id > 1")
        self.assertUnsupported("SELECT DISTINCT id FROM docs")

    def test_join(self):
        self.assertUnsupported("SELECT id FROM docs JOIN other ON docs.id = other.id")
        self.assertUnsupported(
            "SELECT id FROM docs LEFT JOIN other ON docs.id = other.id"
        )
        self.assertUnsupported(
            "SELECT id FROM docs INNER JOIN other ON docs.id = other.id"
        )
        self.assertUnsupported(
            "SELECT id FROM docs CROSS JOIN other ON docs.id = other.id"
        )

    def test_subqueries(self):
        self.assertUnsupported("SELECT id FROM (SELECT id FROM docs)")
        self.assertUnsupported(
            "SELECT id FROM docs WHERE id IN (SELECT id FROM other)"
        )
        self.assertUnsupported("INSERT INTO docs SELECT id FROM other")
        self.assertUnsupported("UPDATE docs SET id = (SELECT id FROM other)")
        self.assertUnsupported("DELETE FROM docs WHERE id IN (SELECT id FROM other)")

    def test_on_duplicate_key_update(self):
        self.assertUnsupported(
            "INSERT INTO docs VALUES (1) ON DUPLICATE KEY UPDATE id = 2"
        )
        self.assertUnsupported("INSERT INTO docs VALUES (1) ON DUPLICATE KEY")

    def test_as_inside_where_operand(self):
        self.assertUnsupported("SELECT id FROM docs WHERE a AS b = 1")

    def test_unterminated_string(self):
        err = self.assertParseError("INSERT INTO docs VALUES ('abc", incomplete=True)
        self.assertIn("unterminated string", str(err))

    def test_unterminated_block_comment(self):
        err = self.assertParseError("SELECT id FROM docs /* oops", incomplete=True)
        self.assertIn("unterminated block comment", str(err))

    def test_truncated_statement_is_incomplete(self):
        self.assertParseError("SELECT", incomplete=True)
        self.assertParseError("SELECT id FROM", incomplete=True)
        self.assertParseError("CREATE TABLE docs (", incomplete=True)
        self.assertParseError("INSERT INTO docs VALUES (", incomplete=True)

    def test_trailing_garbage(self):
        self.assertParseError("SELECT * FROM docs garbage", incomplete=False)
        self.assertParseError("SELECT * FROM docs; SELECT * FROM docs", incomplete=False)
        self.assertParseError("BEGIN; COMMIT", incomplete=False)

    def test_missing_from_is_unsupported_not_incomplete(self):
        err = self.assertUnsupported("SELECT id")
        self.assertFalse(err.incomplete)

    def test_syntax_error_message_has_position(self):
        err = self.assertParseError("SELECT a, * FROM t")
        self.assertIn("position", str(err))

    def test_parse_error_always_has_incomplete_attribute(self):
        for sql in ["", "   ", "FOO", "SELECT", "!!!", "SELECT * FROM"]:
            with self.subTest(sql=sql):
                err = self.assertParseError(sql)
                self.assertIsInstance(err.incomplete, bool)

    def test_only_expected_exception_types(self):
        for sql in [
            "",
            "SELECT",
            "SELECT * FROM t WHERE",
            "INSERT INTO",
            "CREATE TABLE",
            "CREATE TABLE t (",
            "UPDATE",
            "DELETE",
            "UPDATE t SET",
            "SELECT id FROM t ORDER BY",
            "SELECT id FROM t LIMIT",
            "SELECT id FROM t LIMIT -1",
            "SELECT id FROM t LIMIT 'x'",
            "CREATE VECTOR INDEX ON t (c)",
            "SELECT 'x",
            "@@@",
        ]:
            with self.subTest(sql=sql):
                with self.assertRaises((ParseError, UnsupportedError)):
                    parse(sql)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
