"""MiniVectorDB - a small, from-scratch vector database.

The package implements the SQL subset of the OceanBase 2026 competition task
``赛题一：完成一个轻量向量数据库``:

* a column store on plain files with a write ahead log,
* transactions with ``BEGIN`` / ``COMMIT`` / ``ROLLBACK``,
* crash recovery that keeps committed data and discards uncommitted work,
* exact and approximate vector search with ``L2_DISTANCE``.
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["__version__"]
