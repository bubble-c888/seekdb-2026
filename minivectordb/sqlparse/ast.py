"""AST node definitions for the MiniVectorDB SQL subset.

The node set is intentionally small: it mirrors exactly the statement subset
described by the competition task package (see ``mysqltest/README.md``).

Supported statements
--------------------
* ``CREATE TABLE``          -- BIGINT / VARCHAR(n) / VECTOR(n), one BIGINT PK
* ``INSERT``                -- single or multi row, optional column list
* ``SELECT``                -- projections, ``SELECT *``, comparison + AND,
                               ``ORDER BY``, ``LIMIT``
* ``UPDATE`` / ``DELETE``   -- by primary key or by condition
* ``BEGIN`` / ``COMMIT`` / ``ROLLBACK``
* ``CREATE VECTOR INDEX``   -- advanced section of the task statement

Everything outside that subset must be rejected with a parse error so that the
executor never silently executes a statement with a different meaning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Union

# --------------------------------------------------------------------------
# literals
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class IntLiteral:
    """A signed 64-bit integer literal."""

    value: int


@dataclass(frozen=True)
class StrLiteral:
    """A quoted string literal.

    Vector columns store their payload as a string such as ``'[0, 2]'``; the
    coercion to a float32 vector happens in the executor, where the target
    column type is known.
    """

    value: str


Literal = Union[IntLiteral, StrLiteral]


# --------------------------------------------------------------------------
# expressions
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ColumnRef:
    name: str


@dataclass(frozen=True)
class Compare:
    """``left <op> right`` where ``<op>`` is one of ``= < <= > >=``."""

    left: Union[ColumnRef, Literal]
    op: str
    right: Union[ColumnRef, Literal]


@dataclass(frozen=True)
class And:
    left: "Expr"
    right: "Expr"


Expr = Union[ColumnRef, Compare, And]


# --------------------------------------------------------------------------
# DDL
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ColumnDef:
    name: str
    type: str  # "BIGINT" | "VARCHAR" | "VECTOR"
    length: Optional[int] = None  # VARCHAR(n) / VECTOR(n)
    primary_key: bool = False
    not_null: bool = False


@dataclass(frozen=True)
class CreateTable:
    name: str
    columns: List[ColumnDef]
    primary_key: Optional[str] = None


@dataclass(frozen=True)
class CreateVectorIndex:
    name: str
    table: str
    column: str


# --------------------------------------------------------------------------
# DML
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Insert:
    table: str
    columns: Optional[List[str]] = None
    rows: List[List[Literal]] = field(default_factory=list)


@dataclass(frozen=True)
class DistanceFunc:
    """``L2_DISTANCE(<vector column>, '<vector literal>')``."""

    column: str
    vector: str
    name: str = "L2_DISTANCE"


@dataclass(frozen=True)
class SelectItem:
    expr: Union[ColumnRef, DistanceFunc]
    alias: Optional[str] = None


@dataclass(frozen=True)
class OrderItem:
    name: str
    desc: bool = False


@dataclass(frozen=True)
class Select:
    table: str
    items: List[SelectItem]
    where: Optional[Expr] = None
    order_by: List[OrderItem] = field(default_factory=list)
    limit: Optional[int] = None


@dataclass(frozen=True)
class Assignment:
    column: str
    value: Literal


@dataclass(frozen=True)
class Update:
    table: str
    assignments: List[Assignment]
    where: Optional[Expr] = None


@dataclass(frozen=True)
class Delete:
    table: str
    where: Optional[Expr] = None


# --------------------------------------------------------------------------
# transactions
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Begin:
    pass


@dataclass(frozen=True)
class Commit:
    pass


@dataclass(frozen=True)
class Rollback:
    pass


@dataclass(frozen=True)
class SetMode:
    """Switch the retrieval mode.

    Only the small set of variables that the task package uses for the
    exact/approximate switch is accepted; every other ``SET`` statement stays
    unsupported.
    """

    mode: str


Statement = Union[
    CreateTable,
    CreateVectorIndex,
    Insert,
    Select,
    Update,
    Delete,
    Begin,
    Commit,
    Rollback,
    SetMode,
]
