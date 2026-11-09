"""Error hierarchy shared by the parser, the executor and the protocol layer.

Every error carries a MySQL-compatible numeric code and a SQLSTATE so that the
protocol layer can render a stable error object.  The competition statement
requires that unsupported statements are rejected instead of being silently
executed with a different meaning, therefore the default code for an
unsupported construct is ``1235`` (``ER_NOT_SUPPORTED_YET``).
"""

from __future__ import annotations


class MiniVectorDBError(Exception):
    """Base class for every error reported to the client."""

    code = 1105  # ER_UNKNOWN_ERROR
    sqlstate = "HY000"

    def __init__(self, message: str, code: int | None = None, sqlstate: str | None = None):
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        if sqlstate is not None:
            self.sqlstate = sqlstate

    def to_dict(self) -> dict:
        return {
            "error": self.message,
            "code": self.code,
            "sqlstate": self.sqlstate,
        }


class ParseError(MiniVectorDBError):
    """The statement could not be parsed (syntax error or unsupported syntax)."""

    code = 1064  # ER_PARSE_ERROR
    sqlstate = "42000"


class UnsupportedError(MiniVectorDBError):
    """The statement parses but uses a feature outside of the supported subset."""

    code = 1235  # ER_NOT_SUPPORTED_YET
    sqlstate = "42000"


class TableNotFoundError(MiniVectorDBError):
    code = 1146  # ER_NO_SUCH_TABLE
    sqlstate = "42S02"


class ColumnNotFoundError(MiniVectorDBError):
    code = 1054  # ER_BAD_FIELD_ERROR
    sqlstate = "42S22"


class DuplicateKeyError(MiniVectorDBError):
    code = 1062  # ER_DUP_ENTRY
    sqlstate = "23000"


class DataError(MiniVectorDBError):
    """Value could not be represented (vector dimension, NaN, overflow, ...)."""

    code = 11000  # ER_DATA_OUT_OF_RANGE
    sqlstate = "22003"


class ValueTooLongError(MiniVectorDBError):
    code = 1406  # ER_DATA_TOO_LONG
    sqlstate = "22001"


class NotFoundError(MiniVectorDBError):
    """Referenced object (index, table) does not exist."""

    code = 1091  # ER_CANT_DROP_FIELD_OR_KEY
    sqlstate = "42S12"


class TransactionStateError(MiniVectorDBError):
    """Illegal transaction state change, e.g. nested BEGIN."""

    code = 1225  # ER_CANT_EXECUTE_IN_READ_ONLY
    sqlstate = "25000"
