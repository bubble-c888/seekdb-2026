"""Error hierarchy shared by the parser, the executor and the protocol layer.

Every error carries three things:

* a **semantic name** (``duplicate_key``, ``unsupported_sql``,
  ``dimension_mismatch``, ...) -- this is the value the task package expects in
  the ``error`` field of a failed statement;
* a MySQL compatible numeric code and a SQLSTATE, kept for the log output and
  for clients that want them;
* a human readable message.

The task package (2026-competition-round2, ``mysqltest/README.md``) states that
a failed statement is reported with the *error semantic name* and that the
numeric mapping is not frozen, so the wire format only carries the name; the
numeric code and the message are written to stderr instead.
"""

from __future__ import annotations


class MiniVectorDBError(Exception):
    """Base class for every error reported to the client."""

    code = 1105  # ER_UNKNOWN_ERROR
    sqlstate = "HY000"
    name = "internal_error"

    def __init__(self, message: str, code: int | None = None, sqlstate: str | None = None):
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        if sqlstate is not None:
            self.sqlstate = sqlstate

    # -- wire format ----------------------------------------------------
    def to_dict(self) -> dict:
        """The object that is written to stdout for a failed statement.

        The task package compares the ``error`` field with the *error semantic
        name* (``duplicate_key``, ``unsupported_sql``, ``dimension_mismatch``,
        ...) and its expectation objects carry that field only, therefore the
        response carries exactly that field.  The numeric code, the SQLSTATE
        and the message go to stderr, see :meth:`to_log`.
        """
        return {"error": self.name}

    def to_log(self) -> dict:
        """The detailed variant that is written to stderr."""
        return {
            "error": self.name,
            "code": self.code,
            "sqlstate": self.sqlstate,
            "message": self.message,
        }

    def __str__(self) -> str:  # pragma: no cover - debugging helper
        return f"{self.name}: {self.message}"


class ParseError(MiniVectorDBError):
    """The statement could not be parsed (syntax error or unsupported syntax)."""

    code = 1064  # ER_PARSE_ERROR
    sqlstate = "42000"
    name = "syntax_error"


class UnsupportedError(MiniVectorDBError):
    """The statement parses but uses a feature outside of the supported subset."""

    code = 1235  # ER_NOT_SUPPORTED_YET
    sqlstate = "42000"
    name = "unsupported_sql"


class TableNotFoundError(MiniVectorDBError):
    code = 1146  # ER_NO_SUCH_TABLE
    sqlstate = "42S02"
    name = "table_not_found"


class ColumnNotFoundError(MiniVectorDBError):
    code = 1054  # ER_BAD_FIELD_ERROR
    sqlstate = "42S22"
    name = "column_not_found"


class DuplicateKeyError(MiniVectorDBError):
    code = 1062  # ER_DUP_ENTRY
    sqlstate = "23000"
    name = "duplicate_key"


class DataError(MiniVectorDBError):
    """A value could not be represented (NaN, Infinity, overflow, ...)."""

    code = 11000  # ER_DATA_OUT_OF_RANGE
    sqlstate = "22003"
    name = "value_out_of_range"


class DimensionMismatchError(DataError):
    """The vector does not have the dimension declared by its column."""

    code = 11000
    sqlstate = "22003"
    name = "dimension_mismatch"


class ValueTooLongError(MiniVectorDBError):
    code = 1406  # ER_DATA_TOO_LONG
    sqlstate = "22001"
    name = "value_too_long"


class NotFoundError(MiniVectorDBError):
    """Referenced object (index, table) does not exist."""

    code = 1091  # ER_CANT_DROP_FIELD_OR_KEY
    sqlstate = "42S12"
    name = "object_not_found"


class TableExistsError(MiniVectorDBError):
    """A table with the same name already exists."""

    code = 1050  # ER_TABLE_EXISTS_ERROR
    sqlstate = "42S01"
    name = "table_exists"


class IndexExistsError(MiniVectorDBError):
    """An index with the same name already exists on the table."""

    code = 1061  # ER_DUP_KEYNAME
    sqlstate = "42000"
    name = "index_exists"


class TransactionStateError(MiniVectorDBError):
    """Illegal transaction state change, e.g. nested BEGIN."""

    code = 1225  # ER_CANT_EXECUTE_IN_READ_ONLY
    sqlstate = "25000"
    name = "transaction_state_error"
