import copy
import heapq
import json
import math
import operator
import re
import struct
from functools import cmp_to_key
from itertools import islice

from storage import Rows, Storage, read_json
from vector_index import VectorIndex


class SQLError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


TOKEN = re.compile(r"\s*('(?:[^']|'')*'|[A-Za-z_]\w*|-?\d+|<=|>=|<>|!=|[(),;*=<>])")
COMPARISONS = {
    "=": operator.eq, "!=": operator.ne, "<>": operator.ne,
    "<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge,
}


def vector_literal(value, dimension):
    try:
        values = json.loads(value)
    except (TypeError, ValueError) as error:
        raise SQLError("invalid_vector", "Expected a JSON vector") from error
    if not isinstance(values, list):
        raise SQLError("invalid_vector", "Expected a JSON array")
    if len(values) != dimension:
        raise SQLError("dimension_mismatch", "Wrong vector dimension")
    if any(type(item) not in (int, float) for item in values):
        raise SQLError("invalid_vector", "Vector components must be numbers")
    try:
        rounded = [struct.unpack("<f", struct.pack("<f", item))[0] for item in values]
    except (OverflowError, struct.error) as error:
        raise SQLError("invalid_vector", "Vector component outside float32 range") from error
    if not all(map(math.isfinite, rounded)):
        raise SQLError("invalid_vector", "Vector components must be finite")
    return rounded


class Statement:
    def __init__(self, sql):
        sql = sql.strip()
        self.tokens = []
        offset = 0
        while offset < len(sql):
            match = TOKEN.match(sql, offset)
            if not match:
                raise SQLError("syntax_error", "Invalid SQL token")
            self.tokens.append(match[1])
            offset = match.end()
        self.pos = 0

    def take(self):
        if self.pos == len(self.tokens):
            raise SQLError("syntax_error", "Unexpected end of SQL")
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    def accept(self, token):
        if self.pos < len(self.tokens) and self.tokens[self.pos].lower() == token.lower():
            self.pos += 1
            return True
        return False

    def expect(self, token):
        if not self.accept(token):
            raise SQLError("syntax_error", f"Expected {token}")

    def name(self):
        token = self.take()
        if not re.fullmatch(r"[A-Za-z_]\w*", token):
            raise SQLError("syntax_error", "Expected identifier")
        return token.lower()

    def literal(self):
        token = self.take()
        if token.startswith("'"):
            return token[1:-1].replace("''", "'")
        if re.fullmatch(r"-?\d+", token):
            return int(token)
        raise SQLError("syntax_error", "Expected string or integer")

    def finish(self):
        self.accept(";")
        if self.pos != len(self.tokens):
            raise SQLError("syntax_error", "Unexpected SQL suffix")


class Table:
    def __init__(self, columns, primary, rows):
        self.columns = columns
        self.primary = primary
        self.rows = rows
        self.indexes = {}

    def validate(self, column, value):
        if column not in self.columns:
            raise SQLError("unknown_column", column)
        limit = self.columns[column]
        if isinstance(limit, dict):
            return vector_literal(value, limit["dimension"])
        if limit is None:
            if not isinstance(value, int) or not -(2**63) <= value < 2**63:
                raise SQLError("invalid_value", "Expected BIGINT")
        elif not isinstance(value, str) or len(value) > limit:
            raise SQLError("invalid_value", "Value exceeds VARCHAR constraint")
        return value


class Engine:
    def __init__(self, data_dir):
        self.storage = Storage(data_dir)
        self.before_transaction = None
        self.vector_indexes = {}
        legacy = self.storage.root / "tables.json"
        if self.storage.catalog.exists():
            self.load_catalog(read_json(self.storage.catalog))
        elif legacy.exists():
            state = read_json(legacy)
            self.load_catalog(state)
            for name, saved in state.items():
                table = self.tables[name]
                for row in saved["rows"]:
                    table.rows[row[table.primary]] = row
            self.persist()
        else:
            self.tables = {}
        for name, table in self.tables.items():
            for column in set(table.indexes.values()):
                self.get_vector_index(name, table, column)

    def load_catalog(self, state):
        self.tables = {}
        for name, saved in state.items():
            table = Table(saved["columns"], saved["primary"], Rows(self.storage, name))
            table.indexes = saved.get("indexes", {})
            self.tables[name] = table

    def catalog(self):
        return {
            name: {
                "columns": table.columns, "primary": table.primary,
                "indexes": table.indexes,
            }
            for name, table in self.tables.items()
        }

    def persist(self):
        self.storage.commit(self.catalog())

    def get_vector_index(self, name, table, column):
        key = (name, column)
        if key not in self.vector_indexes:
            dimension = table.columns[column]["dimension"]
            # Bound the in-memory tree. Larger tables use streaming exact search.
            limit = min(25000, 2000000 // dimension)
            rows = {}
            text_bytes = 0
            for row_id, row in islice(table.rows.items(), limit + 1):
                text_bytes += sum(4 * len(value) for value in row.values() if isinstance(value, str))
                if len(rows) == limit or text_bytes > 8 * 1024 * 1024:
                    self.vector_indexes[key] = None
                    break
                rows[row_id] = row
            else:
                index = VectorIndex({row_id: row[column] for row_id, row in rows.items()})
                # Keep at most one materialized table cached across queries.
                self.vector_indexes.clear()
                self.vector_indexes[key] = (index, rows)
        return self.vector_indexes[key]

    def begin(self, s):
        s.finish()
        if self.before_transaction is not None:
            raise SQLError("transaction_active", "A transaction is already active")
        self.before_transaction = copy.deepcopy(self.catalog())
        return {}

    def commit(self, s):
        s.finish()
        if self.before_transaction is not None:
            self.persist()
            self.before_transaction = None
        return {}

    def rollback(self, s):
        s.finish()
        self.close()
        return {}

    def close(self):
        if self.before_transaction is not None:
            self.storage.rollback()
            self.load_catalog(self.before_transaction)
            self.before_transaction = None
            self.vector_indexes.clear()

    def table(self, name):
        if name not in self.tables:
            raise SQLError("unknown_table", name)
        return self.tables[name]

    def create(self, s):
        if s.accept("VECTOR"):
            return self.create_vector_index(s)
        s.expect("TABLE")
        name = s.name()
        s.expect("(")
        columns = {}
        primary = None
        while True:
            column = s.name()
            if column in columns:
                raise SQLError("invalid_schema", "Duplicate column")
            if s.accept("BIGINT"):
                columns[column] = None
            else:
                vector = s.accept("VECTOR")
                if vector:
                    if any(isinstance(kind, dict) for kind in columns.values()):
                        raise SQLError("invalid_schema", "Only one vector column is supported")
                else:
                    s.expect("VARCHAR")
                s.expect("(")
                limit = s.literal()
                s.expect(")")
                if not isinstance(limit, int) or limit < 1:
                    raise SQLError("invalid_schema", "Invalid column size")
                columns[column] = {"dimension": limit} if vector else limit
            if s.accept("PRIMARY"):
                s.expect("KEY")
                if primary is not None or columns[column] is not None:
                    raise SQLError("invalid_schema", "Expected one BIGINT primary key")
                primary = column
            if not s.accept(","):
                break
        s.expect(")")
        s.finish()
        if primary is None or name in self.tables:
            raise SQLError("invalid_schema", "Missing primary key or table already exists")
        self.tables[name] = Table(columns, primary, Rows(self.storage, name))
        return {}

    def create_vector_index(self, s):
        s.expect("INDEX")
        name = s.name()
        s.expect("ON")
        table = self.table(s.name())
        s.expect("(")
        column = s.name()
        s.expect(")")
        s.finish()
        if name in table.indexes or not isinstance(table.columns.get(column), dict):
            raise SQLError("invalid_schema", "Expected a new index on a vector column")
        table.indexes[name] = column
        return {}

    def insert(self, s):
        s.expect("INTO")
        table = self.table(s.name())
        s.expect("VALUES")
        pending = {}
        while True:
            s.expect("(")
            row = {}
            for column in table.columns:
                if row:
                    s.expect(",")
                row[column] = table.validate(column, s.literal())
            s.expect(")")
            key = row[table.primary]
            if key in table.rows or key in pending:
                raise SQLError("duplicate_key", "Duplicate primary key")
            pending[key] = row
            if not s.accept(","):
                break
        s.finish()
        for key, row in pending.items():
            table.rows[key] = row
        return {"affected_rows": len(pending)}

    def conditions(self, s, table):
        conditions = []
        if s.accept("WHERE"):
            while True:
                column = s.name()
                comparison = COMPARISONS.get(s.take())
                if comparison is None:
                    raise SQLError("syntax_error", "Expected comparison operator")
                value = table.validate(column, s.literal())
                conditions.append((column, comparison, value))
                if not s.accept("AND"):
                    break
        return conditions

    def matching(self, table, conditions, cached=None):
        rows = table.rows if cached is None else cached
        candidates = rows.items()
        for column, comparison, value in conditions:
            if column == table.primary and comparison is operator.eq:
                candidates = [(value, rows[value])] if value in rows else []
                break
        return (
            (key, row) for key, row in candidates
            if all(compare(row[column], value) for column, compare, value in conditions)
        )

    def select(self, s, search_mode):
        projection = {}
        if not s.accept("*"):
            while True:
                column = s.name()
                if column == "l2_distance" and s.accept("("):
                    vector_column = s.name()
                    s.expect(",")
                    query = s.literal()
                    s.expect(")")
                    alias = s.name() if s.accept("AS") else column
                    projection[alias] = (vector_column, query)
                else:
                    projection[column] = column
                if not s.accept(","):
                    break
        s.expect("FROM")
        table_name = s.name()
        table = self.table(table_name)
        if not projection:
            projection = {column: column for column in table.columns}
        distances = {}
        for alias, expr in projection.items():
            if isinstance(expr, tuple):
                column, query = expr
                if not isinstance(table.columns.get(column), dict):
                    raise SQLError("invalid_vector", "L2_DISTANCE requires a vector column")
                distances[alias] = (column, table.validate(column, query))
            elif expr not in table.columns:
                raise SQLError("unknown_column", expr)
        conditions = self.conditions(s, table)
        order = []
        if s.accept("ORDER"):
            s.expect("BY")
            while True:
                column = s.name()
                if column not in table.columns and column not in distances:
                    raise SQLError("unknown_column", column)
                descending = s.accept("DESC")
                if not descending:
                    s.accept("ASC")
                order.append((column, descending))
                if not s.accept(","):
                    break
        limit = None
        if s.accept("LIMIT"):
            limit = s.literal()
            if not isinstance(limit, int) or limit < 0:
                raise SQLError("invalid_value", "LIMIT must be a nonnegative integer")
        s.finish()
        cached = None
        index = None
        if order and order[0][0] in distances:
            column, query = distances[order[0][0]]
            if column in table.indexes.values():
                entry = self.get_vector_index(table_name, table, column)
                if entry is not None:
                    index, cached = entry
        matches = self.matching(table, conditions, cached)
        if (
            search_mode == "ann" and limit is not None and limit > 0
            and index is not None and len(index.vectors) > 4096
            and len(query) < 32
            and order and order[0][0] in distances and not order[0][1]
            and order[1:] in ([], [(table.primary, False)])
        ):
            allowed = {key for key, _ in matches}
            candidates = index.search(query, allowed, limit) if len(allowed) > limit else allowed
            keys = candidates if len(candidates) == min(len(allowed), limit) else allowed
            matches = ((key, cached[key]) for key in keys)

        def records():
            for key, row in matches:
                values = {alias: math.dist(row[col], query) for alias, (col, query) in distances.items()}
                sort_values = [values[col] if col in values else row[col] for col, _ in order]
                result = [values[alias] if alias in values else row[expr] for alias, expr in projection.items()]
                yield sort_values, key, result

        def compare(left, right):
            for index, (_, descending) in enumerate(order):
                a, b = left[0][index], right[0][index]
                if a != b:
                    return (-1 if a < b else 1) * (-1 if descending else 1)
            return (left[1] > right[1]) - (left[1] < right[1])

        records_iter = records()
        if order and limit is None:
            selected = sorted(records_iter, key=cmp_to_key(compare))
        elif order:
            selected = heapq.nsmallest(limit, records_iter, key=cmp_to_key(compare))
        else:
            selected = islice(records_iter, limit)
        return {"columns": list(projection), "rows": [record[2] for record in selected]}

    def update(self, s):
        table = self.table(s.name())
        s.expect("SET")
        assignments = {}
        while True:
            column = s.name()
            if column == table.primary:
                raise SQLError("unsupported_sql", "Primary key updates are not supported")
            s.expect("=")
            assignments[column] = table.validate(column, s.literal())
            if not s.accept(","):
                break
        conditions = self.conditions(s, table)
        s.finish()
        affected = 0
        with self.storage.matched_keys(self.matching(table, conditions)) as keys:
            for key in keys:
                row = table.rows[key]
                row.update(assignments)
                table.rows[key] = row
                affected += 1
        return {"affected_rows": affected}

    def delete(self, s):
        s.expect("FROM")
        table = self.table(s.name())
        conditions = self.conditions(s, table)
        s.finish()
        affected = 0
        with self.storage.matched_keys(self.matching(table, conditions)) as keys:
            for key in keys:
                del table.rows[key]
                affected += 1
        return {"affected_rows": affected}

    def execute(self, sql, search_mode="exact"):
        statement = Statement(sql)
        operation = statement.take().lower()
        if operation == "select":
            return self.select(statement, search_mode)
        handlers = {
            "create": self.create, "insert": self.insert,
            "update": self.update, "delete": self.delete,
            "begin": self.begin, "commit": self.commit, "rollback": self.rollback,
        }
        if operation not in handlers:
            raise SQLError("unsupported_sql", operation)
        result = handlers[operation](statement)
        if operation in {"create", "insert", "update", "delete"}:
            self.vector_indexes.clear()
            if self.before_transaction is None:
                self.persist()
        return result
