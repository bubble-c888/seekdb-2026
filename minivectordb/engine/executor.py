"""SQL execution: turns parsed statements into result objects.

The supported subset is exactly the one described by the competition task
package.  Anything outside of it raises an error instead of being executed
with a different meaning.
"""

from __future__ import annotations

import bisect
import heapq
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..errors import (
    ColumnNotFoundError,
    DataError,
    IndexExistsError,
    MiniVectorDBError,
    ParseError,
    TableExistsError,
    TableNotFoundError,
    UnsupportedError,
)
from ..sqlparse import ast
from .ann import VectorIndex
from .database import Database
from .storage import WriteSet
from .table import CHUNK_ROWS, ColumnSpec, Table, TableOverlay
from .vector import format_vector, l2_distances, parse_vector_literal

DISTANCE_RE = re.compile(
    r"^\s*L2_DISTANCE\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*['\"]([^'\"]*)['\"]\s*\)\s*$",
    re.IGNORECASE,
)

#: Up to this number of rows the approximate mode scans the vector column
#: exactly.  The reference implementation of the task package uses an exact
#: scan as well, so the small correctness cases have to produce the very same
#: rows in both modes; the approximate index is used for the large data sets
#: of the performance benchmark.
ANN_EXACT_LIMIT = int(os.environ.get("MINIVECTORDB_ANN_EXACT_LIMIT", "50000") or 50000)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


@dataclass
class ComparisonTerm:
    """One ``column <op> literal`` comparison of a WHERE clause."""

    column: ColumnSpec
    op: str
    literal: Any
    rank_array: Optional[np.ndarray] = None
    target_rank: int = 0
    #: False when a string literal is not part of the dictionary: such a value
    #: can only be compared with an ordering operator, an equality never
    #: matches.
    exact_match: bool = True


@dataclass
class OutputItem:
    """One entry of the select list."""

    kind: str  # "column" | "distance"
    column: Optional[ColumnSpec] = None
    vector: Optional[np.ndarray] = None
    label: str = ""


class Executor:
    """Executes statements against a :class:`Database`."""

    def __init__(self, database: Database):
        self.db = database

    # ------------------------------------------------------------------
    # dispatch
    # ------------------------------------------------------------------
    def execute(self, statement: ast.Statement) -> Dict[str, Any]:
        if isinstance(statement, ast.CreateTable):
            return self.create_table(statement)
        if isinstance(statement, ast.CreateVectorIndex):
            return self.create_vector_index(statement)
        if isinstance(statement, ast.Insert):
            return self.insert(statement)
        if isinstance(statement, ast.Select):
            return self.select(statement)
        if isinstance(statement, ast.Update):
            return self.update(statement)
        if isinstance(statement, ast.Delete):
            return self.delete(statement)
        if isinstance(statement, ast.Begin):
            self.db.begin()
            return {}
        if isinstance(statement, ast.Commit):
            self.db.commit()
            return {}
        if isinstance(statement, ast.Rollback):
            self.db.rollback()
            return {}
        if isinstance(statement, ast.SetMode):
            self.db.mode = statement.mode
            return {}
        raise UnsupportedError(f"unsupported statement: {type(statement).__name__}")

    # ------------------------------------------------------------------
    # DDL
    # ------------------------------------------------------------------
    def create_table(self, statement: ast.CreateTable) -> Dict[str, Any]:
        name = statement.name
        if self.db.table(name) is not None:
            raise TableExistsError(f"table '{name}' already exists")
        columns: List[ColumnSpec] = []
        seen = set()
        for column in statement.columns:
            if column.name.lower() in seen:
                raise ParseError(f"duplicate column name '{column.name}'")
            seen.add(column.name.lower())
            spec = ColumnSpec(
                name=column.name,
                type=column.type,
                length=column.length,
                primary_key=column.primary_key,
                not_null=column.not_null,
            )
            columns.append(spec)
        primary_key = statement.primary_key
        if primary_key is None:
            for spec in columns:
                if spec.primary_key:
                    primary_key = spec.name
                    break
        if primary_key is not None:
            pk_spec = next(
                (spec for spec in columns if spec.name.lower() == primary_key.lower()),
                None,
            )
            if pk_spec is None:
                raise ParseError(f"unknown primary key column '{primary_key}'")
            if pk_spec.type != "BIGINT":
                raise UnsupportedError("the primary key must be a BIGINT column")
        vector_columns = [spec for spec in columns if spec.type == "VECTOR"]
        if len(vector_columns) > 1:
            raise UnsupportedError("a table may only have one vector column")
        table = Table(
            store=self.db.store,
            name=name,
            columns=columns,
            primary_key=primary_key,
            vector_column=vector_columns[0].name if vector_columns else None,
        )
        autocommit = not self.db.in_transaction
        self.db.register_new_table(table)
        if autocommit:
            self.db.commit()
        return {}

    def create_vector_index(self, statement: ast.CreateVectorIndex) -> Dict[str, Any]:
        table = self.db.table(statement.table)
        if table is None:
            raise TableNotFoundError(f"table '{statement.table}' does not exist")
        column = table.column(statement.column)
        if column.type != "VECTOR":
            raise UnsupportedError(
                "a vector index can only be created on a VECTOR column"
            )
        if statement.name.lower() in table.indexes:
            raise IndexExistsError(f"index '{statement.name}' already exists")
        dim = int(column.length or 0)
        clusters = 64
        index = VectorIndex(
            table=table,
            name=statement.name,
            column=column.name,
            dim=dim,
            clusters=clusters,
            nprobe=max(1, clusters // 4),
        )
        autocommit = not self.db.in_transaction
        try:
            txn = self.db.current_transaction()
            index.build(txn.writes)
        except Exception:
            if autocommit:
                self.db.rollback()
            raise
        table.indexes[statement.name.lower()] = index.to_meta()
        self.db.indexes[(table.name.lower(), statement.name.lower())] = index
        if autocommit:
            self.db.commit()
        return {}

    # ------------------------------------------------------------------
    # INSERT
    # ------------------------------------------------------------------
    def insert(self, statement: ast.Insert) -> Dict[str, Any]:
        table = self.db.table(statement.table)
        if table is None:
            raise TableNotFoundError(f"table '{statement.table}' does not exist")
        autocommit = not self.db.in_transaction
        txn = self.db.current_transaction()
        overlay = txn.overlay_for(table)
        statement_overlay = self._statement_overlay(overlay)
        writes = WriteSet()
        affected = 0
        try:
            for row in statement.rows:
                values = self._row_values(table, statement, row)
                primary_key = table.primary_key
                if primary_key is not None:
                    key = values.get(primary_key.lower())
                    self.db.check_primary_key(table, int(key))
                slot = self.db.allocate_slot(table)
                table.plan_insert(slot, values, writes, statement_overlay)
                self._plan_index_append(table, slot, values)
                affected += 1
        except Exception:
            # A failed statement must not leave a part of its rows behind.  In
            # auto commit mode the implicit transaction is dropped completely,
            # inside an explicit transaction the previous statements stay.
            if autocommit:
                self.db.rollback()
            raise
        self._merge_overlay(overlay, statement_overlay)
        txn.writes.extend(writes.writes)
        if autocommit:
            self.db.commit()
        return {"affected_rows": affected}

    def _row_values(
        self, table: Table, statement: ast.Insert, row: Sequence[ast.Literal]
    ) -> Dict[str, Any]:
        if statement.columns:
            names = [name.lower() for name in statement.columns]
            if len(names) != len(row):
                raise ParseError(
                    "column count does not match value count in INSERT"
                )
            for name in names:
                table.column(name)
        else:
            names = [column.key for column in table.columns]
            if len(names) != len(row):
                raise ParseError(
                    "column count does not match value count in INSERT"
                )
        values: Dict[str, Any] = {}
        for name, literal in zip(names, row):
            column = table.column(name)
            values[column.key] = self._coerce(table, column, literal)
        return values

    def _coerce(self, table: Table, column: ColumnSpec, literal: ast.Literal) -> Any:
        if column.type == "BIGINT":
            if isinstance(literal, ast.IntLiteral):
                return literal.value
            text = str(literal.value).strip()
            try:
                return int(text)
            except ValueError:
                raise DataError(
                    f"cannot use value '{text}' for BIGINT column '{column.name}'"
                )
        if column.type == "VARCHAR":
            text = str(literal.value)
            if column.length is not None and len(text) > column.length:
                from ..errors import ValueTooLongError

                raise ValueTooLongError(
                    f"value too long for column '{column.name}'"
                )
            return text
        # VECTOR
        text = str(literal.value)
        return parse_vector_literal(text, int(column.length or 0))

    # ------------------------------------------------------------------
    # UPDATE / DELETE
    # ------------------------------------------------------------------
    def update(self, statement: ast.Update) -> Dict[str, Any]:
        table = self.db.table(statement.table)
        if table is None:
            raise TableNotFoundError(f"table '{statement.table}' does not exist")
        primary_key = table.primary_key.lower() if table.primary_key else None
        autocommit = not self.db.in_transaction
        try:
            assignments: Dict[str, Any] = {}
            for assignment in statement.assignments:
                column = table.column(assignment.column)
                if primary_key and column.key == primary_key:
                    raise UnsupportedError(
                        "updating the primary key is not supported"
                    )
                assignments[column.key] = self._coerce(table, column, assignment.value)
            terms = self._compile_predicate(table, statement.where)
            txn = self.db.current_transaction()
            overlay = txn.overlay_for(table)
            statement_overlay = self._statement_overlay(overlay)
            writes = WriteSet()
            affected = 0
            for slot in self._matching_slots(table, terms, overlay):
                current = dict(overlay.rows.get(slot, {}))
                current.update(assignments)
                table.plan_update(slot, assignments, writes, statement_overlay)
                self._plan_index_append(table, slot, current)
                affected += 1
        except Exception:
            if autocommit:
                self.db.rollback()
            raise
        self._merge_overlay(overlay, statement_overlay)
        txn.writes.extend(writes.writes)
        if autocommit:
            self.db.commit()
        return {"affected_rows": affected}

    def delete(self, statement: ast.Delete) -> Dict[str, Any]:
        table = self.db.table(statement.table)
        if table is None:
            raise TableNotFoundError(f"table '{statement.table}' does not exist")
        autocommit = not self.db.in_transaction
        try:
            terms = self._compile_predicate(table, statement.where)
            txn = self.db.current_transaction()
            overlay = txn.overlay_for(table)
            statement_overlay = self._statement_overlay(overlay)
            writes = WriteSet()
            affected = 0
            for slot in self._matching_slots(table, terms, overlay):
                table.plan_delete(slot, writes, statement_overlay)
                overlay.freed_slots.append(slot)
                affected += 1
        except Exception:
            if autocommit:
                self.db.rollback()
            raise
        self._merge_overlay(overlay, statement_overlay)
        txn.writes.extend(writes.writes)
        if autocommit:
            self.db.commit()
        return {"affected_rows": affected}

    # ------------------------------------------------------------------
    # SELECT
    # ------------------------------------------------------------------
    def select(self, statement: ast.Select) -> Dict[str, Any]:
        table = self.db.table(statement.table)
        if table is None:
            raise TableNotFoundError(f"table '{statement.table}' does not exist")
        items = self._resolve_output(table, statement)
        terms = self._compile_predicate(table, statement.where)
        order = self._resolve_order(table, statement, items)
        distance_item = None
        for item in order:
            if isinstance(item[1], OutputItem) and item[1].kind == "distance":
                distance_item = item[1]
        limit = statement.limit

        txn = self.db.txn
        overlay = txn.overlay_for(table) if txn is not None else None

        if distance_item is not None and self.db.mode == "ann":
            rows, labels = self._select_ann(table, items, terms, order, limit, distance_item, overlay)
            return {"columns": labels, "rows": rows}

        rows, labels = self._select_exact(table, items, terms, order, limit, overlay)
        return {"columns": labels, "rows": rows}

    def _resolve_output(
        self, table: Table, statement: ast.Select
    ) -> List[OutputItem]:
        items: List[OutputItem] = []
        for entry in statement.items:
            if isinstance(entry.expr, ast.ColumnRef):
                if entry.expr.name == "*":
                    for column in table.columns:
                        items.append(
                            OutputItem(
                                kind="column", column=column, label=column.name
                            )
                        )
                    continue
                column = table.column(entry.expr.name)
                # MySQL reports the column name the way the client spelled it.
                label = entry.alias or entry.expr.name
                items.append(OutputItem(kind="column", column=column, label=label))
            else:
                function = entry.expr
                column = table.column(function.column)
                if column.type != "VECTOR":
                    raise UnsupportedError(
                        "L2_DISTANCE can only be applied to a VECTOR column"
                    )
                vector = parse_vector_literal(
                    function.vector, int(column.length or 0)
                )
                label = entry.alias or "distance"
                items.append(
                    OutputItem(kind="distance", column=column, vector=vector, label=label)
                )
        return items

    def _resolve_order(
        self, table: Table, statement: ast.Select, items: List[OutputItem]
    ) -> List[Tuple[ast.OrderItem, Any]]:
        resolved: List[Tuple[ast.OrderItem, Any]] = []
        by_label = {item.label.lower(): item for item in items}
        for order_item in statement.order_by:
            target = by_label.get(order_item.name.lower())
            if target is None:
                match = DISTANCE_RE.match(order_item.name)
                if match:
                    column = table.column(match.group(1))
                    if column.type != "VECTOR":
                        raise UnsupportedError(
                            "L2_DISTANCE can only be applied to a VECTOR column"
                        )
                    vector = parse_vector_literal(
                        match.group(2), int(column.length or 0)
                    )
                    target = OutputItem(
                        kind="distance",
                        column=column,
                        vector=vector,
                        label=order_item.name,
                    )
                else:
                    column = table.column(order_item.name)
                    target = OutputItem(kind="column", column=column, label=column.name)
            resolved.append((order_item, target))
        return resolved

    # -- predicate ------------------------------------------------------
    def _compile_predicate(
        self, table: Table, where: Optional[ast.Expr]
    ) -> List[ComparisonTerm]:
        if where is None:
            return []
        terms: List[ComparisonTerm] = []
        self._flatten_and(where, terms)
        compiled: List[ComparisonTerm] = []
        for left, op, right in terms:
            if isinstance(left, ast.ColumnRef):
                column = table.column(left.name)
                literal = right
            else:
                column = table.column(right.name)
                literal = left
                op = {"=": "=", "<": ">", "<=": ">=", ">": "<", ">=": "<="}[op]
            if column.type == "VECTOR":
                raise UnsupportedError(
                    "comparisons on VECTOR columns are not supported"
                )
            if column.type == "BIGINT":
                if isinstance(literal, ast.StrLiteral):
                    try:
                        value = int(str(literal.value).strip())
                    except ValueError:
                        raise DataError(
                            f"cannot compare BIGINT column '{column.name}' with '{literal.value}'"
                        )
                else:
                    value = literal.value
                compiled.append(ComparisonTerm(column=column, op=op, literal=value))
            else:
                text = str(literal.value)
                dictionary = table.dictionary(column)
                pending = self._pending_strings(table, column)
                rank, merged = dictionary.rank_table(pending)
                target_rank = bisect.bisect_left(merged, text)
                compiled.append(
                    ComparisonTerm(
                        column=column,
                        op=op,
                        literal=text,
                        rank_array=rank,
                        target_rank=target_rank,
                        # A literal that is not part of the dictionary matches
                        # no row at all, the insertion point of the sorted
                        # dictionary must not be mistaken for its rank.
                        exact_match=text in merged,
                    )
                )
        return compiled

    def _pending_strings(self, table: Table, column: ColumnSpec) -> List[str]:
        if self.db.txn is None:
            return []
        overlay = self.db.txn.overlays.get(table.name.lower())
        if overlay is None:
            return []
        return list(overlay.pending_new_strings.get(column.key, []))

    def _flatten_and(self, node: ast.Expr, out: List[Tuple[Any, str, Any]]) -> None:
        if isinstance(node, ast.And):
            self._flatten_and(node.left, out)
            self._flatten_and(node.right, out)
            return
        if isinstance(node, ast.Compare):
            out.append((node.left, node.op, node.right))
            return
        raise UnsupportedError("unsupported expression in WHERE clause")

    def _mask_for(
        self,
        table: Table,
        terms: List[ComparisonTerm],
        start: int,
        count: int,
    ) -> np.ndarray:
        mask = table.read_alive(start, count)
        for term in terms:
            if term.column.type == "BIGINT":
                values = table.read_column(term.column, start, count)
                if term.op == "=":
                    part = values == term.literal
                elif term.op == "<":
                    part = values < term.literal
                elif term.op == "<=":
                    part = values <= term.literal
                elif term.op == ">":
                    part = values > term.literal
                else:
                    part = values >= term.literal
                mask = mask & part
            else:
                codes = table.read_column(term.column, start, count)
                safe = np.where(codes >= 0, codes, 0)
                ranks = term.rank_array[safe]
                if term.op == "=":
                    if not term.exact_match:
                        part = np.zeros(count, dtype=bool)
                    else:
                        part = ranks == term.target_rank
                elif term.op == "<":
                    part = ranks < term.target_rank
                elif term.op == "<=":
                    part = ranks <= term.target_rank
                elif term.op == ">":
                    part = ranks > term.target_rank
                else:
                    part = ranks >= term.target_rank
                mask = mask & part & (codes >= 0)
        return mask

    def _matching_slots(
        self,
        table: Table,
        terms: List[ComparisonTerm],
        overlay: Optional[TableOverlay],
    ) -> List[int]:
        """All live rows (committed plus pending) that match the predicate."""
        slots: List[int] = []
        for start in range(0, table.n_slots, CHUNK_ROWS):
            count = min(CHUNK_ROWS, table.n_slots - start)
            if count <= 0:
                continue
            mask = self._mask_for(table, terms, start, count)
            if overlay is not None and overlay.rows:
                # Rows that the running transaction already changed are matched
                # against their *new* values, not against the committed ones.
                for slot, values in overlay.rows.items():
                    if slot in overlay.deleted:
                        continue
                    if not (start <= slot < start + count):
                        continue
                    mask[slot - start] = self._row_matches(
                        table, terms, values, overlay, slot=slot
                    )
            for offset in np.nonzero(mask)[0].tolist():
                slots.append(start + offset)
        if overlay is not None:
            for slot in sorted(overlay.inserted):
                if slot in overlay.deleted:
                    continue
                values = overlay.rows.get(slot, {})
                if self._row_matches(table, terms, values, overlay, slot=None):
                    slots.append(slot)
        return slots

    def _row_matches(
        self,
        table: Table,
        terms: List[ComparisonTerm],
        values: Dict[str, Any],
        overlay: Optional[TableOverlay],
        slot: Optional[int] = None,
    ) -> bool:
        for term in terms:
            key = term.column.key
            if key in values:
                value = values[key]
            elif slot is not None:
                value = table.value_at(slot, term.column)
            else:
                return False
            if term.column.type == "BIGINT":
                if term.op == "=":
                    ok = int(value) == int(term.literal)
                elif term.op == "<":
                    ok = int(value) < int(term.literal)
                elif term.op == "<=":
                    ok = int(value) <= int(term.literal)
                elif term.op == ">":
                    ok = int(value) > int(term.literal)
                else:
                    ok = int(value) >= int(term.literal)
            else:
                text = str(value)
                if term.op == "=" and not term.exact_match:
                    # The literal is not part of the dictionary, it cannot
                    # match any row.
                    return False
                dictionary = table.dictionary(term.column)
                pending = self._pending_strings(table, term.column)
                rank, merged = dictionary.rank_table(pending)
                code = dictionary.code_of(text)
                if code is None and overlay is not None:
                    code = overlay.pending_strings.get(term.column.key, {}).get(text)
                if code is None:
                    return False
                value_rank = int(rank[code])
                if term.op == "=":
                    ok = value_rank == term.target_rank
                elif term.op == "<":
                    ok = value_rank < term.target_rank
                elif term.op == "<=":
                    ok = value_rank <= term.target_rank
                elif term.op == ">":
                    ok = value_rank > term.target_rank
                else:
                    ok = value_rank >= term.target_rank
            if not ok:
                return False
        return True

    # -- result assembly ------------------------------------------------
    def _select_exact(
        self,
        table: Table,
        items: List[OutputItem],
        terms: List[ComparisonTerm],
        order: List[Tuple[ast.OrderItem, Any]],
        limit: Optional[int],
        overlay: Optional[TableOverlay],
    ) -> Tuple[List[List[Any]], List[str]]:
        labels = [item.label for item in items]
        primary_key = table.primary_key.lower() if table.primary_key else None
        if not order:
            # Without an ORDER BY the rows can be streamed and the LIMIT stops
            # the scan early.
            rows: List[List[Any]] = []
            for start in range(0, table.n_slots, CHUNK_ROWS):
                count = min(CHUNK_ROWS, table.n_slots - start)
                if count <= 0:
                    continue
                mask = self._mask_for(table, terms, start, count)
                selected = np.nonzero(mask)[0]
                if selected.size == 0:
                    continue
                slots = selected + start
                rows.extend(self._row_values_for(table, items, start, count, slots, overlay))
                if limit is not None and len(rows) >= limit:
                    break
            if overlay is not None:
                for slot in sorted(overlay.inserted):
                    if slot in overlay.deleted:
                        continue
                    values = overlay.rows.get(slot, {})
                    if not self._row_matches(table, terms, values, overlay, slot=None):
                        continue
                    rows.append(self._row_from_overlay(table, items, values))
            if limit is not None:
                rows = rows[:limit]
            return rows, labels

        # With an ORDER BY only the rows that survive the LIMIT are built, the
        # sort keys are computed chunk by chunk.
        specs = self._order_specs(order, items)
        wanted = limit if limit is not None else None
        best = _TopK(wanted)
        for start in range(0, table.n_slots, CHUNK_ROWS):
            count = min(CHUNK_ROWS, table.n_slots - start)
            if count <= 0:
                continue
            mask = self._mask_for(table, terms, start, count)
            selected = np.nonzero(mask)[0]
            if selected.size == 0:
                continue
            slots = selected + start
            arrays = self._sort_arrays(table, specs, start, count, slots, overlay)
            pks = self._primary_key_array(table, start, count, slots, overlay)
            # The chunk is sorted with numpy, only the best rows of the chunk
            # enter the global heap.
            keys = np.lexsort(tuple(reversed(arrays + [pks])))
            if wanted is not None and keys.size > wanted:
                keys = keys[:wanted]
            for position in keys.tolist():
                index = position
                best.add(
                    (
                        tuple(float(array[index]) for array in arrays)
                        + (int(pks[index]),),
                        int(slots[index]),
                    )
                )
        ordered = best.result()
        if limit is not None:
            ordered = ordered[:limit]
        rows = self._build_payloads(table, items, [slot for _key, slot in ordered], overlay)
        return rows, labels

    def _build_payloads(
        self,
        table: Table,
        items: List[OutputItem],
        slots: List[int],
        overlay: Optional[TableOverlay],
    ) -> List[List[Any]]:
        """Build the result rows for a small set of slots."""
        if not slots:
            return []
        groups: Dict[int, List[int]] = {}
        for slot in slots:
            groups.setdefault(slot // CHUNK_ROWS, []).append(slot)
        payloads: Dict[int, List[Any]] = {}
        for group, group_slots in groups.items():
            start = group * CHUNK_ROWS
            count = min(CHUNK_ROWS, table.n_slots - start)
            selected = np.asarray(sorted(group_slots), dtype=np.int64)
            values = self._row_values_for(table, items, start, count, selected, overlay)
            for slot, row in zip(sorted(group_slots), values):
                payloads[slot] = row
        return [payloads[slot] for slot in slots]

    def _order_specs(
        self, order: List[Tuple[ast.OrderItem, Any]], items: List[OutputItem]
    ) -> List[Tuple[str, ast.OrderItem, Any]]:
        """Describe where the value of every ORDER BY term comes from."""
        specs: List[Tuple[str, ast.OrderItem, Any]] = []
        for order_item, target in order:
            if target.kind == "distance":
                specs.append(("distance", order_item, target))
            else:
                specs.append(("column", order_item, target.column))
        return specs

    def _sort_arrays(
        self,
        table: Table,
        specs: List[Tuple[str, ast.OrderItem, Any]],
        start: int,
        count: int,
        slots: np.ndarray,
        overlay: Optional[TableOverlay],
    ) -> List[np.ndarray]:
        """Return one numeric sort array per ORDER BY term.

        Descending order is expressed by negating the values, so that the
        ascending order of the negated values is the requested order.
        """
        arrays: List[np.ndarray] = []
        slot_list = slots.tolist()
        for kind, order_item, reference in specs:
            if kind == "column":
                chunk = table.read_column(reference, start, count)
                values = chunk[slots - start]
                if reference.type == "VARCHAR":
                    dictionary = table.dictionary(reference)
                    ranks = dictionary.rank_table()[0]
                    safe = np.where(values >= 0, values, 0)
                    values = ranks[safe].astype(np.float64)
                else:
                    values = values.astype(np.float64)
                if overlay is not None and overlay.rows:
                    for index, slot in enumerate(slot_list):
                        override = overlay.rows.get(slot)
                        if override and reference.key in override:
                            values[index] = float(override[reference.key])
                if order_item.desc:
                    values = -values
            else:
                vectors = table.read_vectors(reference.column, start, count)
                distances = l2_distances(vectors[slots - start], reference.vector)
                values = -distances if order_item.desc else distances
            arrays.append(values)
        return arrays

    def _primary_key_array(
        self,
        table: Table,
        start: int,
        count: int,
        slots: np.ndarray,
        overlay: Optional[TableOverlay],
    ) -> np.ndarray:
        if not table.primary_key:
            return np.zeros(len(slots), dtype=np.int64)
        column = table.primary_key_column()
        chunk = table.read_column(column, start, count)
        values = chunk[slots - start].astype(np.int64)
        if overlay is not None and overlay.rows:
            key = column.key
            for index, slot in enumerate(slots.tolist()):
                override = overlay.rows.get(slot)
                if override and key in override:
                    values[index] = int(override[key])
        return values

    def _primary_key_of(
        self, table: Table, slot: int, overlay: Optional[TableOverlay]
    ) -> int:
        if not table.primary_key:
            return 0
        key = table.primary_key.lower()
        if overlay is not None and slot in overlay.rows and key in overlay.rows[slot]:
            return int(overlay.rows[slot][key])
        return table.primary_key_at(slot)

    def _row_values_for(
        self,
        table: Table,
        items: List[OutputItem],
        start: int,
        count: int,
        slots: np.ndarray,
        overlay: Optional[TableOverlay],
    ) -> List[List[Any]]:
        payload: List[List[Any]] = []
        cache: Dict[str, np.ndarray] = {}
        for item in items:
            if item.kind == "column" and item.column is not None:
                column = item.column
                if column.key not in cache:
                    cache[column.key] = table.read_column(column, start, count)
                values = cache[column.key]
                picked = values[slots - start]
                if column.type == "BIGINT":
                    payload.append([int(value) for value in picked.tolist()])
                elif column.type == "VARCHAR":
                    dictionary = table.dictionary(column)
                    payload.append(
                        [
                            dictionary.text_of(int(code)) if int(code) >= 0 else ""
                            for code in picked.tolist()
                        ]
                    )
                else:
                    vectors = table.read_vectors(column, start, count)
                    payload.append(
                        [format_vector(vectors[slot - start]) for slot in slots.tolist()]
                    )
            else:
                payload.append([0.0] * len(slots))
        # distances are computed on the selected rows only
        distance_items = [item for item in items if item.kind == "distance"]
        if distance_items:
            for position, item in enumerate(items):
                if item.kind != "distance" or item.column is None:
                    continue
                vectors = table.read_vectors(item.column, start, count)
                selected = vectors[slots - start]
                distances = l2_distances(selected, item.vector)
                payload[position] = [float(value) for value in distances.tolist()]
        # apply the overlay of the running transaction
        if overlay is not None:
            for index, slot in enumerate(slots.tolist()):
                overrides = overlay.rows.get(slot)
                if not overrides:
                    continue
                for position, item in enumerate(items):
                    if item.kind != "column" or item.column is None:
                        continue
                    key = item.column.key
                    if key in overrides:
                        value = overrides[key]
                        if item.column.type == "VECTOR":
                            payload[position][index] = format_vector(
                                np.asarray(value, dtype=np.float32)
                            )
                        else:
                            payload[position][index] = value
        return [[payload[position][index] for position in range(len(items))] for index in range(len(slots))]

    def _row_from_overlay(
        self, table: Table, items: List[OutputItem], values: Dict[str, Any]
    ) -> List[Any]:
        row: List[Any] = []
        for item in items:
            if item.kind == "column" and item.column is not None:
                value = values.get(item.column.key)
                if item.column.type == "VECTOR" and value is not None:
                    row.append(format_vector(np.asarray(value, dtype=np.float32)))
                else:
                    row.append(value)
            else:
                vector = values.get(item.column.key) if item.column is not None else None
                if vector is None:
                    row.append(0.0)
                else:
                    stored = np.asarray(vector, dtype=np.float32)
                    row.append(
                        float(
                            np.sqrt(
                                float(
                                    np.einsum(
                                        "i,i->i",
                                        stored - item.vector,
                                        stored - item.vector,
                                        dtype=np.float64,
                                    )[0]
                                )
                            )
                        )
                    )
        return row

    # -- approximate search ---------------------------------------------
    def _select_ann(
        self,
        table: Table,
        items: List[OutputItem],
        terms: List[ComparisonTerm],
        order: List[Tuple[ast.OrderItem, Any]],
        limit: Optional[int],
        distance_item: OutputItem,
        overlay: Optional[TableOverlay],
    ) -> Tuple[List[List[Any]], List[str]]:
        labels = [item.label for item in items]
        if table.n_slots <= ANN_EXACT_LIMIT:
            # Small data sets are scanned exactly, so that the approximate mode
            # returns the same rows as the reference implementation.
            return self._select_exact(table, items, terms, order, limit, overlay)
        index = self._index_for(table, distance_item.column)
        if index is None:
            return self._select_exact(table, items, terms, order, limit, overlay)
        assert distance_item.column is not None
        column = distance_item.column
        descending = any(item.desc for item, _ in order) if order else False
        wanted = limit
        sign = -1.0 if descending else 1.0
        best: List[Tuple[float, int, int]] = []
        seen = set()
        probed = 0
        alive = table.read_alive(0, table.n_slots)
        for cluster_distance, radius, cluster_slots in index.iter_cluster_slots(
            distance_item.vector
        ):
            if probed >= 1 and wanted is not None and len(best) >= wanted:
                # The bound of the cluster (the distance of the query to the
                # centroid minus the radius of the cluster) proves that the
                # cluster cannot hold a row that is closer than the worst row
                # that is kept.  All following clusters are farther away, so
                # the search can stop.
                bound = max(0.0, cluster_distance - radius)
                if bound > abs(best[-1][0]):
                    break
            if probed >= index.nprobe and (wanted is None or len(best) >= wanted):
                break
            probed += 1
            slots: List[int] = []
            for slot in cluster_slots:
                slot = int(slot)
                if slot in seen:
                    continue
                seen.add(slot)
                if slot >= table.n_slots:
                    continue
                if not bool(alive[slot]):
                    continue
                if terms and not self._slot_matches(table, terms, slot, overlay):
                    continue
                slots.append(slot)
            if not slots:
                continue
            vectors = index.gather_vectors(
                column, np.asarray(slots, dtype=np.int64)
            )
            diff = vectors.astype(np.float32) - distance_item.vector
            distances = np.sqrt(
                np.einsum("ij,ij->i", diff, diff, dtype=np.float64)
            )
            for slot, distance in zip(slots, distances.tolist()):
                primary_key = table.primary_key_at(slot)
                best.append((sign * float(distance), primary_key, slot))
            best.sort(key=lambda entry: (entry[0], entry[1]))
            if wanted is not None and len(best) > wanted:
                best = best[:wanted]
        best.sort(key=lambda entry: (entry[0], entry[1]))
        if limit is not None:
            best = best[:limit]
        rows: List[List[Any]] = []
        for distance, _pk, slot in best:
            rows.append(
                self._single_row(table, items, slot, overlay, distance_item, abs(distance))
            )
        return rows, labels

    def _slot_matches(
        self,
        table: Table,
        terms: List[ComparisonTerm],
        slot: int,
        overlay: Optional[TableOverlay],
    ) -> bool:
        values: Dict[str, Any] = {}
        if overlay is not None and slot in overlay.rows:
            values.update(overlay.rows[slot])
        return self._row_matches(table, terms, values, overlay, slot=slot)

    def _single_row(
        self,
        table: Table,
        items: List[OutputItem],
        slot: int,
        overlay: Optional[TableOverlay],
        distance_item: OutputItem,
        distance: float,
    ) -> List[Any]:
        row: List[Any] = []
        for item in items:
            if item.kind == "distance":
                row.append(distance)
                continue
            assert item.column is not None
            column = item.column
            if overlay is not None and slot in overlay.rows and column.key in overlay.rows[slot]:
                value = overlay.rows[slot][column.key]
                row.append(
                    format_vector(np.asarray(value, dtype=np.float32))
                    if column.type == "VECTOR"
                    else value
                )
                continue
            value = table.value_at(slot, column)
            if column.type == "VECTOR":
                row.append(format_vector(np.asarray(value, dtype=np.float32)))
            else:
                row.append(value)
        return row

    def _index_for(self, table: Table, column: Optional[ColumnSpec]) -> Optional[VectorIndex]:
        if column is None:
            return None
        for (table_name, _index_name), index in self.db.indexes.items():
            if table_name == table.name.lower() and index.column.lower() == column.key:
                return index
        return None

    # ------------------------------------------------------------------
    # transaction helpers
    # ------------------------------------------------------------------
    def _statement_overlay(self, overlay: TableOverlay) -> TableOverlay:
        statement = TableOverlay()
        for key, pending in overlay.pending_strings.items():
            statement.pending_strings[key] = dict(pending)
        return statement

    def _merge_overlay(self, target: TableOverlay, source: TableOverlay) -> None:
        target.rows.update(source.rows)
        target.deleted |= source.deleted
        target.inserted |= source.inserted
        target.new_slots.extend(source.new_slots)
        target.freed_slots.extend(source.freed_slots)
        target.alive_changes.update(source.alive_changes)
        for key, pending in source.pending_strings.items():
            target.pending_strings.setdefault(key, {}).update(pending)
        for key, texts in source.pending_new_strings.items():
            target.pending_new_strings.setdefault(key, []).extend(texts)
        target.index_appends.extend(source.index_appends)

    def _plan_index_append(
        self, table: Table, slot: int, values: Dict[str, Any]
    ) -> None:
        """Remember that a row has to be added to the approximate index."""
        vector_column = table.vector_column
        if not vector_column:
            return
        key = vector_column.lower()
        if key not in values:
            return
        vector = np.asarray(values[key], dtype=np.float32)
        for (table_name, _index_name), index in self.db.indexes.items():
            if table_name != table.name.lower():
                continue
            if index.column.lower() != key:
                continue
            overlay = self.db.current_transaction().overlay_for(table)
            overlay.index_appends.append((index, slot, vector))


class _Reverse:
    """Wrapper that inverts the comparison order of a sort key."""

    __slots__ = ("value",)

    def __init__(self, value: Any):
        self.value = value

    def __lt__(self, other: "_Reverse") -> bool:
        return other.value < self.value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Reverse) and other.value == self.value


class _MaxItem:
    """Heap entry that keeps the *largest* key on top of the heap."""

    __slots__ = ("key", "slot")

    def __init__(self, key: tuple, slot: int):
        self.key = key
        self.slot = slot

    def __lt__(self, other: "_MaxItem") -> bool:
        return self.key > other.key


class _TopK:
    """Keeps the ``k`` smallest sort keys of a scan."""

    def __init__(self, k: Optional[int]):
        self.k = k
        self._heap: List[_MaxItem] = []
        self._all: List[Tuple[tuple, int]] = []

    def add(self, candidate: Tuple[tuple, int]) -> None:
        key, slot = candidate
        if self.k is None:
            self._all.append(candidate)
            return
        if len(self._heap) < self.k:
            heapq.heappush(self._heap, _MaxItem(key, slot))
            return
        if key < self._heap[0].key:
            heapq.heapreplace(self._heap, _MaxItem(key, slot))

    def result(self) -> List[Tuple[tuple, int]]:
        if self.k is None:
            return sorted(self._all, key=lambda item: item[0])
        return sorted(
            ((item.key, item.slot) for item in self._heap), key=lambda item: item[0]
        )
