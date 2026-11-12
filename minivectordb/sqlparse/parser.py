"""Recursive descent parser for the MiniVectorDB SQL subset.

The parser consumes the token stream produced by
:mod:`minivectordb.sqlparse.lexer` and builds the AST defined in
:mod:`minivectordb.sqlparse.ast`.

Design notes
------------
* Keywords are *not* distinguished from identifiers by the lexer.  The parser
  decides from the context whether a word is a keyword or a name, therefore a
  column may legitimately be called ``set`` or ``from``.
* Identifiers keep their original spelling in the AST (so that the emitted
  column labels match what the user wrote).  Whenever the parser itself has
  to compare two identifiers it folds them to lower case first.
* Every error raised by :func:`parse` is either a :class:`ParseError` (code
  1064) or an :class:`UnsupportedError` (code 1235).  Both carry the dynamic
  attribute ``incomplete``: ``True`` means "the text ended in the middle of a
  token or statement, read more input and retry", ``False`` means "this is a
  definite error".
"""

from __future__ import annotations

from typing import List, Optional

from ..errors import ParseError, UnsupportedError
from . import ast
from .lexer import Token, tokenize

__all__ = ["parse", "Parser"]


#: variables that the task package may use to switch between the exact and the
#: approximate retrieval.  Every other ``SET`` statement stays unsupported.
MODE_VARIABLES = frozenset(
    {
        "search_mode",
        "mode",
        "vector_search_mode",
        "retrieval_mode",
        "minivectordb.search_mode",
        "minivectordb.mode",
        "minivectordb_mode",
    }
)


#: statements that are recognised but explicitly outside of the supported
#: subset -- they must be reported as *unsupported* and not as a syntax error
UNSUPPORTED_STATEMENT_KEYWORDS = frozenset(
    {
        "ALTER",
        "DROP",
        "TRUNCATE",
        "RENAME",
        "SET",
        "SHOW",
        "DESCRIBE",
        "EXPLAIN",
        "USE",
        "GRANT",
        "REVOKE",
        "WITH",
        "UNION",
        "INTERSECT",
        "EXCEPT",
        "REPLACE",
        "CALL",
        "PREPARE",
        "EXECUTE",
        "SAVEPOINT",
        "LOCK",
        "ANALYZE",
        "OPTIMIZE",
        "PROCEDURE",
        "START",
        "END",
    }
)

#: words that are rejected as *unsupported* when they show up where the
#: grammar does not allow them
UNSUPPORTED_KEYWORDS = frozenset(
    {
        "ALTER",
        "DROP",
        "TRUNCATE",
        "RENAME",
        "SET",
        "SHOW",
        "DESCRIBE",
        "EXPLAIN",
        "USE",
        "GRANT",
        "REVOKE",
        "WITH",
        "UNION",
        "INTERSECT",
        "EXCEPT",
        "REPLACE",
        "CALL",
        "PREPARE",
        "EXECUTE",
        "SAVEPOINT",
        "LOCK",
        "ANALYZE",
        "OPTIMIZE",
        "PROCEDURE",
        "START",
        "END",
        "DISTINCT",
        "GROUP",
        "HAVING",
        "JOIN",
        "INNER",
        "LEFT",
        "RIGHT",
        "FULL",
        "CROSS",
        "OUTER",
        "ON",
        "DUPLICATE",
        "OR",
        "IN",
        "BETWEEN",
        "LIKE",
        "IS",
        "NOT",
        "NULL",
        "INDEX",
        "DATABASE",
        "SCHEMA",
        "VIEW",
        "UNIQUE",
    }
)

#: aggregate functions that are outside of the supported subset
AGGREGATE_FUNCTIONS = frozenset({"COUNT", "SUM", "AVG", "MIN", "MAX"})

#: comparison operators of the supported subset
COMPARISON_OPS = frozenset(("=", "<", "<=", ">", ">="))

#: comparison operators that are valid SQL but outside of the subset
OTHER_COMPARISON_OPS = frozenset(("!=", "<>"))

#: column types of the supported subset
COLUMN_TYPES = frozenset(("BIGINT", "VARCHAR", "VECTOR"))


def _unsupported(detail: str):
    """Raise an :class:`UnsupportedError` with the ``incomplete`` attribute."""
    err = UnsupportedError(f"unsupported: {detail}")
    err.incomplete = False
    raise err


class Parser:
    """Recursive descent parser over the token stream."""

    def __init__(self, tokens: List[Token], sql: str = "") -> None:
        self.tokens = tokens
        self.sql = sql
        self.i = 0

    # ------------------------------------------------------------------
    # token helpers
    # ------------------------------------------------------------------
    def peek(self, offset: int = 0) -> Token:
        idx = self.i + offset
        if idx >= len(self.tokens):
            idx = len(self.tokens) - 1
        if idx < 0:
            idx = 0
        return self.tokens[idx]

    def advance(self) -> Token:
        tok = self.tokens[self.i]
        if tok.kind != "eof":
            self.i += 1
        return tok

    def at_op(self, symbol: str) -> bool:
        return self.peek().is_op(symbol)

    def at_word(self, word: str) -> bool:
        return self.peek().is_word() and self.peek().upper() == word

    def at_eof(self) -> bool:
        return self.peek().kind == "eof"

    # ------------------------------------------------------------------
    # errors
    # ------------------------------------------------------------------
    def _err(self, detail: str, tok: Optional[Token] = None, incomplete: bool = False):
        tok = tok if tok is not None else self.peek()
        err = ParseError(f"syntax error at position {tok.pos}: {detail}")
        err.incomplete = incomplete
        raise err

    def _missing(self, what: str):
        """The input ended before a required token was found."""
        tok = self.peek()
        err = ParseError(
            f"syntax error at position {tok.pos}: expected {what} "
            f"but reached end of input"
        )
        err.incomplete = True
        raise err

    def _unexpected(self, what: str, tok: Optional[Token] = None):
        """Report an unexpected token, honouring the unsupported word list."""
        tok = tok if tok is not None else self.peek()
        if tok.kind == "word":
            up = tok.upper()
            if up in UNSUPPORTED_KEYWORDS:
                _unsupported(f"{up} is not supported")
        self._err(f"expected {what}", tok)

    def _expect_op(self, symbol: str) -> Token:
        tok = self.peek()
        if tok.is_op(symbol):
            return self.advance()
        if tok.kind == "eof":
            self._missing(f"'{symbol}'")
        self._err(f"expected '{symbol}'", tok)

    def _expect_word(self, word: str) -> Token:
        tok = self.peek()
        if tok.is_word() and tok.upper() == word:
            return self.advance()
        if tok.kind == "eof":
            self._missing(word)
        self._err(f"expected {word}", tok)

    def _expect_name(self, what: str) -> str:
        tok = self.peek()
        if tok.kind == "word":
            self.advance()
            return tok.value
        if tok.kind == "eof":
            self._missing(what)
        if tok.kind == "op" and tok.value == "(":
            nxt = self.peek(1)
            if nxt.is_word() and nxt.upper() == "SELECT":
                _unsupported("subqueries are not supported")
        self._err(f"expected {what}", tok)

    def _expect_int(self, what: str) -> int:
        tok = self.peek()
        if tok.kind == "int":
            self.advance()
            return tok.value
        if tok.kind == "eof":
            self._missing(what)
        self._err(f"expected {what}", tok)

    # ------------------------------------------------------------------
    # entry point
    # ------------------------------------------------------------------
    def parse(self) -> ast.Statement:
        stmt = self.parse_statement()
        if self.at_op(";"):
            self.advance()
        tok = self.peek()
        if tok.kind != "eof":
            self._trailing_error(tok)
        return stmt

    def _trailing_error(self, tok: Token):
        if tok.kind == "word":
            up = tok.upper()
            if up in UNSUPPORTED_KEYWORDS:
                _unsupported(f"{up} is not supported")
        self._err(f"unexpected trailing input {tok.text!r}", tok)

    def parse_statement(self) -> ast.Statement:
        tok = self.peek()
        if tok.kind == "eof":
            self._err("empty statement", tok)
        if tok.kind != "word":
            self._err("expected a statement", tok)
        kw = tok.upper()
        if kw == "CREATE":
            return self.parse_create()
        if kw == "INSERT":
            return self.parse_insert()
        if kw == "SELECT":
            return self.parse_select()
        if kw == "UPDATE":
            return self.parse_update()
        if kw == "DELETE":
            return self.parse_delete()
        if kw == "BEGIN":
            self.advance()
            return ast.Begin()
        if kw == "COMMIT":
            self.advance()
            return ast.Commit()
        if kw == "ROLLBACK":
            self.advance()
            return ast.Rollback()
        if kw == "SET":
            return self.parse_set_mode()
        if kw in UNSUPPORTED_STATEMENT_KEYWORDS:
            _unsupported(f"{kw} is not supported")
        self._err(f"unknown statement {tok.text!r}", tok)

    def parse_set_mode(self) -> ast.Statement:
        """Parse ``SET <mode variable> = 'exact'|'ann'``.

        The task package switches between the exact and the approximate
        retrieval with a mode parameter; only that small set of variables is
        accepted, every other ``SET`` statement stays unsupported.
        """
        self.advance()  # SET
        if self.at_word("SESSION") or self.at_word("GLOBAL") or self.at_word("LOCAL"):
            self.advance()
        name = self._expect_name("a variable name")
        key = name.lower()
        if key.startswith("@@"):
            key = key[2:]
        if key not in MODE_VARIABLES:
            _unsupported(f"SET {name} is not supported")
        if not self.at_op("="):
            tok = self.peek()
            if tok.kind == "eof":
                self._missing("=")
            self._err("expected '=' in SET statement", tok)
        self.advance()  # '='
        tok = self.peek()
        if tok.kind == "string":
            self.advance()
            value = tok.value.lower()
        elif tok.kind == "word":
            self.advance()
            value = tok.value.lower()
        else:
            self._err("expected a mode value in SET statement", tok)
        if value not in ("exact", "ann"):
            self._err(f"unknown retrieval mode {value!r}", tok)
        return ast.SetMode(mode=value)

    # ------------------------------------------------------------------
    # CREATE
    # ------------------------------------------------------------------
    def parse_create(self) -> ast.Statement:
        self.advance()  # CREATE
        tok = self.peek()
        if tok.kind != "word":
            if tok.kind == "eof":
                self._missing("a statement")
            self._err("expected TABLE, VECTOR INDEX or UNIQUE INDEX", tok)
        kw = tok.upper()
        if kw == "TABLE":
            return self.parse_create_table()
        if kw == "VECTOR":
            self.advance()
            if not self.at_word("INDEX"):
                tok2 = self.peek()
                if tok2.kind == "eof":
                    self._missing("INDEX")
                self._err("expected INDEX after CREATE VECTOR", tok2)
            self.advance()  # INDEX
            return self.parse_create_vector_index()
        if kw in ("INDEX", "UNIQUE", "DATABASE", "SCHEMA", "VIEW"):
            _unsupported(f"CREATE {kw} is not supported")
        _unsupported(f"CREATE {kw} is not supported")

    def parse_create_table(self) -> ast.CreateTable:
        self.advance()  # TABLE
        name = self._expect_name("a table name")
        self._expect_op("(")
        columns: List[ast.ColumnDef] = []
        inline_pk: Optional[str] = None
        table_pk: Optional[str] = None
        while True:
            if self.at_word("PRIMARY") and self.peek(1).is_word() and self.peek(1).upper() == "KEY":
                self.advance()  # PRIMARY
                self.advance()  # KEY
                self._expect_op("(")
                pk_name = self._expect_name("a column name")
                self._expect_op(")")
                if table_pk is not None:
                    self._err("duplicate PRIMARY KEY definition")
                table_pk = pk_name
            elif self.at_word("FOREIGN") or self.at_word("CONSTRAINT"):
                # The task statement lists foreign keys as not supported; they
                # have to be reported as an unsupported statement, not as a
                # malformed one.
                _unsupported("foreign key constraints are not supported")
            elif self.at_word("CHECK"):
                _unsupported("check constraints are not supported")
            elif self.at_word("UNIQUE"):
                _unsupported("unique constraints are not supported")
            else:
                col = self._parse_column_def()
                columns.append(col)
                if col.primary_key:
                    if inline_pk is not None:
                        self._err("duplicate PRIMARY KEY definition")
                    inline_pk = col.name
            if self.at_op(","):
                self.advance()
                continue
            break
        self._expect_op(")")

        primary_key = inline_pk or table_pk
        if table_pk is not None:
            folded = table_pk.lower()
            found = None
            for idx, col in enumerate(columns):
                if col.name.lower() == folded:
                    found = idx
                    break
            if found is None:
                self._err(f"unknown column {table_pk!r} in PRIMARY KEY clause")
            columns[found] = ast.ColumnDef(
                name=columns[found].name,
                type=columns[found].type,
                length=columns[found].length,
                primary_key=True,
                not_null=columns[found].not_null,
            )
            primary_key = columns[found].name
        return ast.CreateTable(name=name, columns=columns, primary_key=primary_key)

    def _parse_column_def(self) -> ast.ColumnDef:
        name = self._expect_name("a column name")
        tok = self.peek()
        if tok.kind != "word":
            if tok.kind == "eof":
                self._missing("a column type")
            self._err("expected a column type", tok)
        type_name = tok.upper()
        if type_name not in COLUMN_TYPES:
            self._err(f"unknown column type {tok.text!r}", tok)
        self.advance()
        length: Optional[int] = None
        if type_name in ("VARCHAR", "VECTOR"):
            self._expect_op("(")
            length = self._expect_int("a length")
            self._expect_op(")")
        not_null = False
        primary_key = False
        while True:
            if self.at_word("NOT"):
                self.advance()
                self._expect_word("NULL")
                not_null = True
            elif self.at_word("PRIMARY"):
                self.advance()
                self._expect_word("KEY")
                primary_key = True
            elif self.at_word("REFERENCES"):
                # An inline foreign key, e.g. ``p BIGINT REFERENCES t(id)``.
                _unsupported("foreign key constraints are not supported")
            elif self.at_word("UNIQUE"):
                _unsupported("unique constraints are not supported")
            else:
                break
        return ast.ColumnDef(
            name=name,
            type=type_name,
            length=length,
            primary_key=primary_key,
            not_null=not_null,
        )

    def parse_create_vector_index(self) -> ast.CreateVectorIndex:
        name = self._expect_name("an index name")
        self._expect_word("ON")
        table = self._expect_name("a table name")
        self._expect_op("(")
        column = self._expect_name("a column name")
        self._expect_op(")")
        return ast.CreateVectorIndex(name=name, table=table, column=column)

    # ------------------------------------------------------------------
    # INSERT
    # ------------------------------------------------------------------
    def parse_insert(self) -> ast.Insert:
        self.advance()  # INSERT
        self._expect_word("INTO")
        table = self._expect_name("a table name")
        columns: Optional[List[str]] = None
        if self.at_op("("):
            self.advance()
            cols: List[str] = []
            while True:
                cols.append(self._expect_name("a column name"))
                if self.at_op(","):
                    self.advance()
                    continue
                break
            self._expect_op(")")
            columns = cols
        if self.at_word("SELECT"):
            _unsupported("INSERT ... SELECT is not supported")
        if self.at_word("SET"):
            _unsupported("INSERT ... SET is not supported")
        self._expect_word("VALUES")
        rows: List[List[ast.Literal]] = []
        while True:
            self._expect_op("(")
            values: List[ast.Literal] = []
            while True:
                values.append(self._parse_value())
                if self.at_op(","):
                    self.advance()
                    continue
                break
            self._expect_op(")")
            rows.append(values)
            if self.at_op(","):
                self.advance()
                continue
            break
        if self.at_word("ON"):
            _unsupported("ON DUPLICATE KEY UPDATE is not supported")
        return ast.Insert(table=table, columns=columns, rows=rows)

    def _parse_value(self) -> ast.Literal:
        tok = self.peek()
        if tok.kind == "int":
            self.advance()
            return ast.IntLiteral(tok.value)
        if tok.kind == "string":
            self.advance()
            return ast.StrLiteral(tok.value)
        if tok.kind == "op" and tok.value == "(":
            _unsupported("subqueries are not supported")
        if tok.kind == "word" and tok.upper() in ("NULL", "NOT", "DEFAULT"):
            _unsupported(f"{tok.upper()} is not supported")
        if tok.kind == "eof":
            self._missing("a literal value")
        self._err("expected a literal value", tok)

    # ------------------------------------------------------------------
    # SELECT
    # ------------------------------------------------------------------
    def parse_select(self) -> ast.Select:
        self.advance()  # SELECT
        if self.at_word("DISTINCT"):
            _unsupported("SELECT DISTINCT is not supported")
        items: List[ast.SelectItem] = []
        if self.at_op("*"):
            self.advance()
            items.append(ast.SelectItem(expr=ast.ColumnRef("*"), alias=None))
            if self.at_op(","):
                self._err("'*' cannot be combined with other select items")
        else:
            while True:
                items.append(self._parse_select_item())
                if self.at_op(","):
                    self.advance()
                    continue
                break
        if not self.at_word("FROM"):
            _unsupported("SELECT without FROM is not supported")
        self.advance()  # FROM
        table = self._expect_name("a table name")

        where: Optional[ast.Expr] = None
        order_by: List[ast.OrderItem] = []
        limit: Optional[int] = None
        if self.at_word("WHERE"):
            self.advance()
            where = self._parse_expr()
        if self.at_word("ORDER"):
            self.advance()
            self._expect_word("BY")
            while True:
                order_by.append(self._parse_order_item())
                if self.at_op(","):
                    self.advance()
                    continue
                break
        limit: Optional[int] = None
        offset = 0
        if self.at_word("LIMIT"):
            self.advance()
            first = self._parse_limit_number()
            if self.at_op(","):
                # ``LIMIT offset, count``
                self.advance()
                offset = first
                limit = self._parse_limit_number()
            elif self.at_word("OFFSET"):
                # ``LIMIT count OFFSET offset``
                self.advance()
                limit = first
                offset = self._parse_limit_number()
            else:
                limit = first
        return ast.Select(
            table=table,
            items=items,
            where=where,
            order_by=order_by,
            limit=limit,
            offset=offset,
        )

    def _parse_limit_number(self) -> int:
        """A non-negative number of rows."""
        tok = self.peek()
        if tok.kind != "int":
            if tok.kind == "eof":
                self._missing("a LIMIT value")
            self._err("LIMIT expects a non-negative integer", tok)
        if tok.value < 0:
            self._err("LIMIT expects a non-negative integer", tok)
        self.advance()
        return int(tok.value)

    def _parse_select_item(self) -> ast.SelectItem:
        tok = self.peek()
        if tok.kind == "op" and tok.value == "*":
            self._err("'*' cannot be combined with other select items", tok)
        if tok.kind != "word":
            if tok.kind == "eof":
                self._missing("a select item")
            if tok.kind in ("int", "string"):
                # A projection that is a literal (``SELECT 1``) is outside the
                # subset of the task statement.
                _unsupported("a literal select item is not supported")
            self._err("expected a select item", tok)
        if self.peek(1).is_op("("):
            return self._parse_function_item()
        self.advance()
        name = tok.value
        return ast.SelectItem(expr=ast.ColumnRef(name), alias=self._parse_alias())

    def _parse_function_item(self) -> ast.SelectItem:
        tok = self.advance()  # function name
        fname = tok.upper()
        if fname == "L2_DISTANCE":
            self._expect_op("(")
            column = self._expect_name("a column name")
            self._expect_op(",")
            vtok = self.peek()
            if vtok.kind != "string":
                if vtok.kind == "eof":
                    self._missing("a vector literal")
                self._err("L2_DISTANCE expects a quoted vector literal", vtok)
            self.advance()
            self._expect_op(")")
            expr = ast.DistanceFunc(column=column, vector=vtok.value)
        elif fname in AGGREGATE_FUNCTIONS:
            _unsupported(f"the aggregate function {fname}() is not supported")
        else:
            _unsupported(f"the function {fname}() is not supported")
        return ast.SelectItem(expr=expr, alias=self._parse_alias())

    def _parse_alias(self) -> Optional[str]:
        if self.at_word("AS"):
            self.advance()
            return self._expect_name("an alias")
        return None

    def _parse_order_item(self) -> ast.OrderItem:
        tok = self.peek()
        if tok.kind == "word" and self.peek(1).is_op("("):
            # ORDER BY L2_DISTANCE(embedding, '[0, 0]') -- the distance is
            # ordered by without being part of the select list.
            name = self._parse_order_function()
        else:
            name = self._expect_name("a column name")
        desc = False
        if self.at_word("DESC"):
            self.advance()
            desc = True
        elif self.at_word("ASC"):
            self.advance()
        return ast.OrderItem(name=name, desc=desc)

    def _parse_order_function(self) -> str:
        """Parse a distance function in an ORDER BY clause.

        The raw text of the call is kept, the executor resolves it against the
        columns of the table.
        """
        start = self.peek().pos
        tok = self.advance()  # function name
        fname = tok.upper()
        if fname != "L2_DISTANCE":
            _unsupported(f"the function {fname}() cannot be used in ORDER BY")
        self._expect_op("(")
        self._expect_name("a column name")
        self._expect_op(",")
        vtok = self.peek()
        if vtok.kind != "string":
            if vtok.kind == "eof":
                self._missing("a vector literal")
            self._err("L2_DISTANCE expects a quoted vector literal", vtok)
        self.advance()
        close = self._expect_op(")")
        return self.sql[start:close.pos + len(close.text)]

    # ------------------------------------------------------------------
    # expressions
    # ------------------------------------------------------------------
    def _parse_expr(self) -> ast.Expr:
        left = self._parse_comparison()
        while True:
            if self.at_word("OR"):
                _unsupported("OR is not supported")
            if not self.at_word("AND"):
                break
            self.advance()
            right = self._parse_comparison()
            left = ast.And(left=left, right=right)
        return left

    def _parse_comparison(self) -> ast.Expr:
        left = self._parse_operand()
        tok = self.peek()
        if tok.kind == "op":
            if tok.value in COMPARISON_OPS:
                op = tok.value
                self.advance()
            elif tok.value in OTHER_COMPARISON_OPS:
                _unsupported(f"the operator {tok.value!r} is not supported")
            else:
                self._err("expected a comparison operator", tok)
        elif tok.kind == "word" and tok.upper() in ("IN", "BETWEEN", "LIKE", "IS", "AS"):
            _unsupported(f"{tok.upper()} is not supported")
        elif tok.kind == "eof":
            self._missing("a comparison operator")
        else:
            self._err("expected a comparison operator", tok)
        right = self._parse_operand()
        return ast.Compare(left=left, op=op, right=right)

    def _parse_operand(self):
        tok = self.peek()
        if tok.kind == "int":
            self.advance()
            return ast.IntLiteral(tok.value)
        if tok.kind == "string":
            self.advance()
            return ast.StrLiteral(tok.value)
        if tok.kind == "op" and tok.value == "(":
            nxt = self.peek(1)
            if nxt.is_word() and nxt.upper() == "SELECT":
                _unsupported("subqueries are not supported")
            self._err("expected a column name or a literal", tok)
        if tok.kind == "word":
            up = tok.upper()
            if up in ("NULL", "NOT", "IN", "BETWEEN", "LIKE", "IS", "OR", "AND"):
                _unsupported(f"{up} is not supported")
            self.advance()
            return ast.ColumnRef(tok.value)
        if tok.kind == "eof":
            self._missing("a column name or a literal")
        self._err("expected a column name or a literal", tok)

    # ------------------------------------------------------------------
    # UPDATE / DELETE
    # ------------------------------------------------------------------
    def parse_update(self) -> ast.Update:
        self.advance()  # UPDATE
        table = self._expect_name("a table name")
        if not self.at_word("SET"):
            tok = self.peek()
            if tok.kind == "eof":
                self._missing("SET")
            if tok.is_word() and tok.upper() in UNSUPPORTED_KEYWORDS:
                _unsupported(f"{tok.upper()} is not supported")
            self._err("expected SET", tok)
        self.advance()  # SET
        assignments: List[ast.Assignment] = []
        while True:
            column = self._expect_name("a column name")
            self._expect_op("=")
            assignments.append(ast.Assignment(column=column, value=self._parse_value()))
            if self.at_op(","):
                self.advance()
                continue
            break
        where: Optional[ast.Expr] = None
        if self.at_word("WHERE"):
            self.advance()
            where = self._parse_expr()
        return ast.Update(table=table, assignments=assignments, where=where)

    def parse_delete(self) -> ast.Delete:
        self.advance()  # DELETE
        self._expect_word("FROM")
        table = self._expect_name("a table name")
        where: Optional[ast.Expr] = None
        if self.at_word("WHERE"):
            self.advance()
            where = self._parse_expr()
        return ast.Delete(table=table, where=where)


def parse(sql: str) -> ast.Statement:
    """Parse exactly one SQL statement of the supported subset.

    The trailing ``;`` is optional.  A :class:`~minivectordb.errors.ParseError`
    is raised for a syntax error, an
    :class:`~minivectordb.errors.UnsupportedError` for a construct that is
    outside of the supported subset.

    Both exceptions carry the attribute ``incomplete`` (a bool).  ``True``
    means that the input text ended in the middle of a token or statement, so
    the caller may read more input and retry.
    """
    if not isinstance(sql, str):
        err = ParseError("syntax error: the statement must be a string")
        err.incomplete = False
        raise err
    try:
        tokens = tokenize(sql)
        parser = Parser(tokens, sql)
        return parser.parse()
    except (ParseError, UnsupportedError) as exc:
        if not hasattr(exc, "incomplete"):
            exc.incomplete = False
        raise
    except RecursionError:
        err = ParseError("syntax error: the statement is nested too deeply")
        err.incomplete = False
        raise err
    except Exception as exc:  # pragma: no cover - defensive, never expected
        err = ParseError(f"syntax error: the statement could not be parsed ({exc})")
        err.incomplete = False
        raise err from exc
