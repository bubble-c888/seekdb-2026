"""Table level storage.

A table is a set of fixed width column files plus a bitmap of live rows.  All
mutations are expressed as *plans*: the executor decides what changes, the
table turns them into byte patches that are only applied to the data files
when the surrounding transaction commits.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..errors import (
    ColumnNotFoundError,
    DataError,
    DuplicateKeyError,
    ValueTooLongError,
)
from .storage import FileStore, WriteSet

CHUNK_ROWS = 8192

#: Upper bound for the in-memory column cache.  Columns are only cached when
#: they fit into the budget, so that a table that is larger than the memory
#: limit is still read from disk in chunks.
CACHE_BUDGET = int(os.environ.get("MINIVECTORDB_CACHE_BYTES", str(32 * 1024 * 1024)))

#: Process wide column cache: ``{cache key: numpy array}``.
_COLUMN_CACHE: Dict[str, np.ndarray] = {}
_COLUMN_CACHE_BYTES = 0


def _cache_key(store: FileStore, key: str) -> str:
    return f"{store.root}|{key}"


def cache_invalidate(store: FileStore, key: str) -> None:
    """Drop a cached column after the file behind it changed."""
    global _COLUMN_CACHE_BYTES
    cache_key = _cache_key(store, key)
    array = _COLUMN_CACHE.pop(cache_key, None)
    if array is not None:
        _COLUMN_CACHE_BYTES -= array.nbytes


def cache_store(store: FileStore, key: str, array: np.ndarray) -> np.ndarray:
    """Put a column into the cache when it fits into the budget."""
    global _COLUMN_CACHE_BYTES
    if array.nbytes > CACHE_BUDGET:
        return array
    cache_key = _cache_key(store, key)
    if cache_key not in _COLUMN_CACHE:
        if _COLUMN_CACHE_BYTES + array.nbytes > CACHE_BUDGET:
            _COLUMN_CACHE.clear()
            _COLUMN_CACHE_BYTES = 0
        _COLUMN_CACHE[cache_key] = array
        _COLUMN_CACHE_BYTES += array.nbytes
    return _COLUMN_CACHE[cache_key]


@dataclass
class ColumnSpec:
    name: str
    type: str  # BIGINT | VARCHAR | VECTOR
    length: Optional[int] = None
    primary_key: bool = False
    not_null: bool = False

    @property
    def key(self) -> str:
        return self.name.lower()

    def to_meta(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "length": self.length,
            "primary_key": self.primary_key,
            "not_null": self.not_null,
        }

    @classmethod
    def from_meta(cls, payload: Dict[str, Any]) -> "ColumnSpec":
        return cls(
            name=payload["name"],
            type=payload["type"],
            length=payload.get("length"),
            primary_key=bool(payload.get("primary_key", False)),
            not_null=bool(payload.get("not_null", False)),
        )


def _safe(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)


class StringDictionary:
    """Dictionary encoding for a ``VARCHAR`` column.

    The committed dictionary is persisted in two files: the string blobs and an
    index holding the offset of every code.  New strings that are introduced by
    a transaction stay *pending* until that transaction commits, so an aborted
    transaction never leaves a code in the column files that has no blob.
    """

    def __init__(self, store: FileStore, data_key: str, index_key: str):
        self.store = store
        self.data_key = data_key
        self.index_key = index_key
        self.strings: List[str] = []
        self.codes: Dict[str, int] = {}
        self._rank: Optional[np.ndarray] = None
        self._sorted: List[str] = []
        self.load()

    # -- persistence ----------------------------------------------------
    def load(self) -> None:
        self.strings = []
        self.codes = {}
        offsets = self.store.read_all(self.index_key)
        if not offsets:
            self._dirty()
            return
        count = len(offsets) // 8
        blob = self.store.read_all(self.data_key)
        for code in range(count):
            offset = struct.unpack_from("<Q", offsets, code * 8)[0]
            length = struct.unpack_from("<I", blob, offset)[0]
            raw = blob[offset + 4 : offset + 4 + length]
            text = raw.decode("utf-8")
            self.strings.append(text)
            self.codes[text] = code
        self._dirty()

    # -- lookups --------------------------------------------------------
    def __len__(self) -> int:
        return len(self.strings)

    def code_of(self, text: str) -> Optional[int]:
        return self.codes.get(text)

    def text_of(self, code: int) -> str:
        return self.strings[code]

    def _dirty(self) -> None:
        self._rank = None
        self._sorted = []

    def rank_table(self, extra: Sequence[str] = ()) -> Tuple[np.ndarray, List[str]]:
        """Return ``(rank per code, sorted strings)``.

        ``extra`` holds strings that are only visible inside the current
        transaction; they take part in the ordering so that comparisons stay
        consistent with the rows that the transaction inserted.
        """
        if self._rank is None or extra:
            merged = sorted(set(self.strings) | set(extra))
            rank = np.zeros(len(self.strings) + len(extra), dtype=np.int32)
            for position, text in enumerate(merged):
                code = self.codes.get(text)
                if code is not None:
                    rank[code] = position
                else:
                    offset = len(self.strings) + list(extra).index(text)
                    rank[offset] = position
            if extra:
                return rank, merged
            self._rank = rank
            self._sorted = merged
            return rank, merged
        return self._rank, self._sorted

    def pending_code(self, text: str, pending: Dict[str, int]) -> int:
        """Return a provisional code for a string that is not committed yet."""
        if text in self.codes:
            return self.codes[text]
        if text in pending:
            return pending[text]
        code = len(self.strings) + len(pending)
        pending[text] = code
        return code

    def plan_append(self, text: str, writes: WriteSet) -> None:
        """Append a string to the dictionary (deferred to the commit)."""
        blob = text.encode("utf-8")
        offset = writes.logical_size(self.data_key, self.store.size(self.data_key))
        writes.add(self.data_key, offset, struct.pack("<I", len(blob)) + blob)
        index_offset = writes.logical_size(
            self.index_key, self.store.size(self.index_key)
        )
        writes.add(self.index_key, index_offset, struct.pack("<Q", offset))
        self.strings.append(text)
        self.codes[text] = len(self.strings) - 1
        self._dirty()


class PrimaryKeyIndex:
    """In-memory open addressing hash index from primary key to row slot.

    The index is rebuilt from the primary key column when the database starts,
    therefore it does not need its own file and it can never disagree with the
    committed data.
    """

    def __init__(self, capacity: int = 1024):
        self.capacity = max(16, 1 << (max(16, capacity) - 1).bit_length())
        self.keys = np.zeros(self.capacity, dtype=np.int64)
        self.slots = np.zeros(self.capacity, dtype=np.int64)
        self.count = 0

    @staticmethod
    def _hash(key: int, mask: int) -> int:
        value = (key * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        value ^= value >> 29
        value = (value * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        value ^= value >> 32
        return int(value) & mask

    def lookup(self, key: int) -> Optional[int]:
        mask = self.capacity - 1
        position = self._hash(key, mask)
        while True:
            value = int(self.slots[position])
            if value == 0:
                return None
            if value > 0 and int(self.keys[position]) == key:
                return value - 1
            position = (position + 1) & mask

    def insert(self, key: int, slot: int) -> None:
        if (self.count + 1) * 10 >= self.capacity * 7:
            self._grow()
        mask = self.capacity - 1
        position = self._hash(key, mask)
        while True:
            value = int(self.slots[position])
            if value == 0 or value == -1:
                self.keys[position] = key
                self.slots[position] = slot + 1
                self.count += 1
                return
            position = (position + 1) & mask

    def remove(self, key: int) -> None:
        mask = self.capacity - 1
        position = self._hash(key, mask)
        while True:
            value = int(self.slots[position])
            if value == 0:
                return
            if value > 0 and int(self.keys[position]) == key:
                self.slots[position] = -1
                self.count -= 1
                return
            position = (position + 1) & mask

    def _grow(self) -> None:
        old_keys, old_slots = self.keys, self.slots
        self.capacity *= 2
        self.keys = np.zeros(self.capacity, dtype=np.int64)
        self.slots = np.zeros(self.capacity, dtype=np.int64)
        self.count = 0
        mask = self.capacity - 1
        for position in range(old_slots.size):
            slot = int(old_slots[position])
            if slot <= 0:
                continue
            key = int(old_keys[position])
            target = self._hash(key, mask)
            while int(self.slots[target]) != 0:
                target = (target + 1) & mask
            self.keys[target] = key
            self.slots[target] = slot
            self.count += 1


@dataclass
class TableOverlay:
    """Rows changed by a transaction that is not committed yet."""

    rows: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    deleted: set = field(default_factory=set)
    new_slots: List[int] = field(default_factory=list)
    freed_slots: List[int] = field(default_factory=list)
    pending_strings: Dict[str, Dict[str, int]] = field(default_factory=dict)
    pending_new_strings: Dict[str, List[str]] = field(default_factory=dict)
    inserted: set = field(default_factory=set)
    alive_changes: Dict[int, bool] = field(default_factory=dict)
    index_appends: List[Tuple[Any, int, Any]] = field(default_factory=list)


class Table:
    """One table of the database."""

    def __init__(
        self,
        store: FileStore,
        name: str,
        columns: List[ColumnSpec],
        primary_key: Optional[str],
        vector_column: Optional[str],
        n_slots: int = 0,
        free_slots: Optional[List[int]] = None,
    ):
        self.store = store
        self.name = name
        self.columns = columns
        self.primary_key = primary_key
        self.vector_column = vector_column
        self.n_slots = n_slots
        self.free_slots: List[int] = list(free_slots or [])
        self.index: Dict[str, int] = {col.key: pos for pos, col in enumerate(columns)}
        self.dictionaries: Dict[str, StringDictionary] = {}
        self.pk_index = PrimaryKeyIndex()
        self.indexes: Dict[str, Dict[str, Any]] = {}
        # The alive bitmap is kept in memory; it mirrors ``alive.bits`` on disk.
        self.alive = bytearray()

    # -- metadata -------------------------------------------------------
    def to_meta(self) -> Dict[str, Any]:
        return {
            "columns": [col.to_meta() for col in self.columns],
            "primary_key": self.primary_key,
            "vector_column": self.vector_column,
            "n_slots": self.n_slots,
            "free_slots": list(self.free_slots),
            "indexes": self.indexes,
        }

    @classmethod
    def from_meta(cls, store: FileStore, name: str, meta: Dict[str, Any]) -> "Table":
        table = cls(
            store=store,
            name=name,
            columns=[ColumnSpec.from_meta(item) for item in meta["columns"]],
            primary_key=meta.get("primary_key"),
            vector_column=meta.get("vector_column"),
            n_slots=int(meta.get("n_slots", 0)),
            free_slots=list(meta.get("free_slots", [])),
        )
        table.indexes = dict(meta.get("indexes", {}))
        return table

    # -- file keys ------------------------------------------------------
    def column_key(self, column: ColumnSpec) -> str:
        return f"t/{self.name}/col_{_safe(column.name)}.dat"

    @property
    def alive_key(self) -> str:
        return f"t/{self.name}/alive.bits"

    def dictionary_keys(self, column: ColumnSpec) -> Tuple[str, str]:
        base = f"t/{self.name}/dict_{_safe(column.name)}"
        return base + ".dat", base + ".idx"

    def dictionary(self, column: ColumnSpec) -> StringDictionary:
        cached = self.dictionaries.get(column.key)
        if cached is None:
            data_key, index_key = self.dictionary_keys(column)
            cached = StringDictionary(self.store, data_key, index_key)
            self.dictionaries[column.key] = cached
        return cached

    # -- column helpers -------------------------------------------------
    def column(self, name: str) -> ColumnSpec:
        position = self.index.get(name.lower())
        if position is None:
            raise ColumnNotFoundError(f"unknown column '{name}'")
        return self.columns[position]

    def primary_key_column(self) -> ColumnSpec:
        if not self.primary_key:
            raise ColumnNotFoundError(f"table '{self.name}' has no primary key")
        return self.column(self.primary_key)

    def vector_spec(self) -> Optional[ColumnSpec]:
        if not self.vector_column:
            return None
        return self.column(self.vector_column)

    def dtype_of(self, column: ColumnSpec):
        if column.type == "BIGINT":
            return np.int64
        if column.type == "VARCHAR":
            return np.int32
        return np.float32

    def itemsize(self, column: ColumnSpec) -> int:
        if column.type == "BIGINT":
            return 8
        if column.type == "VARCHAR":
            return 4
        return 4 * int(column.length or 0)

    # -- reading --------------------------------------------------------
    def _full_column(self, column: ColumnSpec) -> Optional[np.ndarray]:
        """Return the whole column from the cache, loading it when it fits."""
        key = self.column_key(column)
        cache_key = _cache_key(self.store, key)
        cached = _COLUMN_CACHE.get(cache_key)
        if cached is not None:
            return cached
        per_row = self.itemsize(column)
        if max(self.n_slots, 1) * per_row > CACHE_BUDGET:
            return None
        rows = max(self.n_slots, 1)
        if column.type == "VECTOR":
            array = self._read_vectors_from_disk(column, 0, rows)
        else:
            array = self._read_column_from_disk(column, 0, rows)
        return cache_store(self.store, key, array)

    def read_column(self, column: ColumnSpec, start: int, count: int) -> np.ndarray:
        if count <= 0:
            return np.zeros(0, dtype=self.dtype_of(column))
        full = self._full_column(column)
        if full is not None:
            return self._slice(full, start, count, self.dtype_of(column))
        return self._read_column_from_disk(column, start, count)

    def read_vectors(self, column: ColumnSpec, start: int, count: int) -> np.ndarray:
        dim = int(column.length or 0)
        if count <= 0 or dim == 0:
            return np.zeros((0, dim), dtype=np.float32)
        full = self._full_column(column)
        if full is not None:
            return self._slice(full, start, count, np.float32)
        return self._read_vectors_from_disk(column, start, count)

    @staticmethod
    def _slice(array: np.ndarray, start: int, count: int, dtype) -> np.ndarray:
        """Return ``count`` values starting at ``start``, padded with zeros."""
        if array.ndim == 1:
            available = max(0, array.shape[0] - start)
            if available >= count:
                return array[start : start + count]
            out = np.zeros(count, dtype=dtype)
            if available:
                out[:available] = array[start : start + available]
            return out
        rows = array.shape[0]
        available = max(0, rows - start)
        if available >= count:
            return array[start : start + count]
        out = np.zeros((count, array.shape[1]), dtype=dtype)
        if available:
            out[:available] = array[start : start + available]
        return out

    def _read_column_from_disk(self, column: ColumnSpec, start: int, count: int) -> np.ndarray:
        key = self.column_key(column)
        path = self.store.path(key)
        try:
            with open(path, "rb") as handle:
                handle.seek(start * self.itemsize(column))
                data = np.fromfile(handle, dtype=self.dtype_of(column), count=count)
        except FileNotFoundError:
            return np.zeros(count, dtype=self.dtype_of(column))
        if data.size < count:
            padded = np.zeros(count, dtype=self.dtype_of(column))
            padded[: data.size] = data
            return padded
        return data

    def _read_vectors_from_disk(self, column: ColumnSpec, start: int, count: int) -> np.ndarray:
        dim = int(column.length or 0)
        path = self.store.path(self.column_key(column))
        try:
            with open(path, "rb") as handle:
                handle.seek(start * dim * 4)
                data = np.fromfile(handle, dtype=np.float32, count=count * dim)
        except FileNotFoundError:
            return np.zeros((count, dim), dtype=np.float32)
        if data.size < count * dim:
            padded = np.zeros(count * dim, dtype=np.float32)
            padded[: data.size] = data
            data = padded
        return data.reshape(count, dim)

    def read_alive(self, start: int, count: int) -> np.ndarray:
        if count <= 0:
            return np.zeros(0, dtype=np.bool_)
        needed = (start + count + 7) // 8
        if len(self.alive) < needed:
            bits = np.zeros(start + count, dtype=np.uint8)
        else:
            bits = np.unpackbits(np.frombuffer(self.alive, dtype=np.uint8), bitorder="little")
        if bits.size < start + count:
            bits = np.concatenate([bits, np.zeros(start + count - bits.size, dtype=np.uint8)])
        return bits[start : start + count].astype(np.bool_)

    def load_alive(self) -> None:
        """Read the alive bitmap from disk into memory."""
        self.alive = bytearray(self.store.read_all(self.alive_key))

    def live_slots(self) -> np.ndarray:
        alive = self.read_alive(0, self.n_slots)
        return np.nonzero(alive)[0]

    def value_at(self, slot: int, column: ColumnSpec) -> Any:
        if column.type == "BIGINT":
            raw = self.store.read(self.column_key(column), slot * 8, 8)
            return int(struct.unpack("<q", raw)[0]) if len(raw) == 8 else 0
        if column.type == "VARCHAR":
            raw = self.store.read(self.column_key(column), slot * 4, 4)
            code = int(struct.unpack("<i", raw)[0]) if len(raw) == 4 else -1
            dictionary = self.dictionary(column)
            if 0 <= code < len(dictionary):
                return dictionary.text_of(code)
            return ""
        dim = int(column.length or 0)
        raw = self.store.read(self.column_key(column), slot * dim * 4, dim * 4)
        if len(raw) < dim * 4:
            raw = raw + b"\x00" * (dim * 4 - len(raw))
        return np.frombuffer(raw, dtype=np.float32).copy()

    def primary_key_at(self, slot: int) -> int:
        column = self.primary_key_column()
        raw = self.store.read(self.column_key(column), slot * 8, 8)
        if len(raw) < 8:
            return 0
        return int(struct.unpack("<q", raw)[0])

    # -- mutation planning ---------------------------------------------
    def plan_insert(
        self,
        slot: int,
        values: Dict[str, Any],
        writes: WriteSet,
        overlay: TableOverlay,
    ) -> None:
        """Produce the byte patches that materialise a new row."""
        for column in self.columns:
            value = values.get(column.key)
            if column.type == "BIGINT":
                writes.add(self.column_key(column), slot * 8, struct.pack("<q", int(value)))
            elif column.type == "VARCHAR":
                text = str(value)
                if column.length is not None and len(text) > column.length:
                    raise ValueTooLongError(
                        f"value too long for column '{column.name}'"
                    )
                code = self._plan_string_code(column, text, overlay)
                writes.add(self.column_key(column), slot * 4, struct.pack("<i", code))
            else:
                vector = np.asarray(value, dtype=np.float32)
                writes.add(
                    self.column_key(column),
                    slot * self.itemsize(column),
                    vector.tobytes(),
                )
        self.plan_set_alive(slot, True, writes, overlay)
        overlay.rows[slot] = dict(values)
        overlay.inserted.add(slot)

    def plan_update(
        self,
        slot: int,
        values: Dict[str, Any],
        writes: WriteSet,
        overlay: TableOverlay,
    ) -> None:
        for column in self.columns:
            if column.key not in values:
                continue
            value = values[column.key]
            if column.type == "BIGINT":
                writes.add(self.column_key(column), slot * 8, struct.pack("<q", int(value)))
            elif column.type == "VARCHAR":
                text = str(value)
                if column.length is not None and len(text) > column.length:
                    raise ValueTooLongError(
                        f"value too long for column '{column.name}'"
                    )
                code = self._plan_string_code(column, text, overlay)
                writes.add(self.column_key(column), slot * 4, struct.pack("<i", code))
            else:
                vector = np.asarray(value, dtype=np.float32)
                writes.add(
                    self.column_key(column),
                    slot * self.itemsize(column),
                    vector.tobytes(),
                )
        overlay.rows.setdefault(slot, {}).update(values)

    def plan_delete(self, slot: int, writes: WriteSet, overlay: TableOverlay) -> None:
        self.plan_set_alive(slot, False, writes, overlay)
        overlay.deleted.add(slot)
        overlay.rows.pop(slot, None)

    def _plan_string_code(
        self, column: ColumnSpec, text: str, overlay: TableOverlay
    ) -> int:
        """Return the dictionary code for ``text``.

        Codes of strings that the transaction introduces are provisional: they
        are handed out in the same order in which the strings are appended to
        the dictionary when the transaction commits.
        """
        dictionary = self.dictionary(column)
        code = dictionary.code_of(text)
        if code is not None:
            return code
        pending = overlay.pending_strings.setdefault(column.key, {})
        if text in pending:
            return pending[text]
        code = len(dictionary) + len(pending)
        pending[text] = code
        overlay.pending_new_strings.setdefault(column.key, []).append(text)
        return code

    def plan_set_alive(
        self, slot: int, alive: bool, writes: WriteSet, overlay: TableOverlay
    ) -> None:
        """Patch the alive bitmap.

        The byte that carries the bit is rebuilt from the committed file plus
        every pending change of the transaction, so that several rows of the
        same byte do not overwrite each other.
        """
        byte_index, bit_index = divmod(slot, 8)
        current = self.alive[byte_index] if byte_index < len(self.alive) else 0
        for other, state in overlay.alive_changes.items():
            other_byte, other_bit = divmod(other, 8)
            if other_byte != byte_index:
                continue
            if state:
                current |= 1 << other_bit
            else:
                current &= ~(1 << other_bit) & 0xFF
        if alive:
            current |= 1 << bit_index
        else:
            current &= ~(1 << bit_index) & 0xFF
        writes.add(self.alive_key, byte_index, bytes([current]))
        overlay.alive_changes[slot] = alive

    # -- committed state ------------------------------------------------
    def apply_alive(self, slot: int, alive: bool) -> None:
        byte_index, bit_index = divmod(slot, 8)
        if byte_index >= len(self.alive):
            self.alive.extend(b"\x00" * (byte_index + 1 - len(self.alive)))
        if alive:
            self.alive[byte_index] |= 1 << bit_index
        else:
            self.alive[byte_index] &= ~(1 << bit_index) & 0xFF

    def rebuild_primary_key_index(self) -> None:
        self.pk_index = PrimaryKeyIndex(max(16, self.n_slots * 2))
        if not self.primary_key:
            return
        column = self.primary_key_column()
        chunk_size = CHUNK_ROWS
        for start in range(0, self.n_slots, chunk_size):
            count = min(chunk_size, self.n_slots - start)
            values = self.read_column(column, start, count)
            alive = self.read_alive(start, count)
            for offset in np.nonzero(alive)[0]:
                self.pk_index.insert(int(values[offset]), start + int(offset))

    def check_duplicate(self, key: int) -> None:
        if self.pk_index.lookup(key) is not None:
            raise DuplicateKeyError(f"duplicate entry for primary key {key}")
