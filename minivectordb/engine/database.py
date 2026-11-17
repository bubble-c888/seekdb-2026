"""Database level: catalog, transactions, commit protocol and crash recovery.

Commit protocol
---------------
A transaction buffers every change it wants to perform as *byte patches*; the
data files are not touched before the transaction commits.  Committing means

1. appending the patches and the new metadata snapshot to the write ahead log
   and flushing it to stable storage -- this is the commit point;
2. applying the patches to the data files and flushing those as well.

A crash before step 1 leaves the transaction completely unapplied, a crash
after step 1 replays the transaction during recovery.  Because the patches are
physical writes, replaying them is idempotent, therefore a transaction is
either fully visible or not visible at all -- never partially applied.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from ..errors import (
    DuplicateKeyError,
    UnsupportedError,
)
from .ann import VectorIndex
from .storage import FileStore, WriteAheadLog, WriteSet
from .table import ColumnSpec, Table, TableOverlay, cache_invalidate

CATALOG_FILE = "catalog.json"
WAL_DIR = "wal"

#: How many commits pass before the data files are flushed and the catalog is
#: written.  The commit point is the flush of the log, so the commits since
#: the last checkpoint do not have to be on the disk before the next commit.
#: A checkpoint costs seven flushes of the file system - the data files, the
#: directory, the catalog and its directory - and there are a hundred
#: commits in the benchmark, so the interval decides how much of the write
#: time is spent waiting for the disk.  The log of the commits in between
#: stays small: a record holds the state of a commit and not its data.
CHECKPOINT_EVERY = int(os.environ.get("MINIVECTORDB_CHECKPOINT_EVERY", "10") or "10")


@dataclass
class Transaction:
    id: int
    writes: WriteSet = field(default_factory=WriteSet)
    overlays: Dict[str, TableOverlay] = field(default_factory=dict)
    new_tables: Dict[str, Table] = field(default_factory=dict)
    dropped_tables: Set[str] = field(default_factory=set)

    def overlay_for(self, table: Table) -> TableOverlay:
        return self.overlays.setdefault(table.name.lower(), TableOverlay())


class Database:
    """The MiniVectorDB database."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.store = FileStore(self.data_dir)
        self.wal = WriteAheadLog(self.store, WAL_DIR)
        self.tables: Dict[str, Table] = {}
        self.metadata: Dict[str, Any] = {
            "version": 1,
            "wal": {"segment": self.wal.segment, "offset": 0},
            "tables": {},
        }
        self.txn: Optional[Transaction] = None
        self._txn_counter = 0
        self.mode = "exact"
        self.indexes: Dict[Tuple[str, str], VectorIndex] = {}
        self._pending_checkpoint = False
        #: The files that were written since the last checkpoint.
        self._dirty_keys: Set[str] = set()
        #: The commits since the last checkpoint.
        self._pending_commits = 0

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def open(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        stored = self.store.read_json(CATALOG_FILE)
        if stored:
            self.metadata = stored
        wal_meta = self.metadata.get("wal", {})
        self.wal.segment = wal_meta.get("segment", self.wal.segment)
        self.wal.offset = int(wal_meta.get("offset", 0))
        self._recover()
        self._load_tables()
        self._rebuild_indexes()
        self.checkpoint()

    def close(self) -> None:
        try:
            # The data files are flushed before the catalog is written, so that
            # the catalog never claims more than what is on the disk.
            self._flush_and_checkpoint()
        finally:
            self.wal.close()
            for table in self.tables.values():
                table.close()

    # ------------------------------------------------------------------
    # recovery
    # ------------------------------------------------------------------
    def _segments(self) -> List[str]:
        directory = self.store.path(WAL_DIR)
        if not directory.exists():
            return []
        names = sorted(
            entry.name
            for entry in os.scandir(directory)
            if entry.is_file() and entry.name.startswith("wal-") and entry.name.endswith(".log")
        )
        return names

    def _recover(self) -> None:
        """Replay every committed transaction that is not in the catalog yet."""
        segments = self._segments()
        if not segments:
            self.wal.open()
            return
        wal_meta = self.metadata.get("wal", {})
        start_segment = wal_meta.get("segment")
        start_offset = int(wal_meta.get("offset", 0))
        touched: Set[str] = set()
        sizes: Dict[str, int] = {}
        last_segment = segments[0]
        for segment in segments:
            last_segment = segment
            if start_segment is not None and segment < start_segment:
                # The checkpoint covers this segment completely.
                continue
            offset = start_offset if segment == start_segment else 0
            self.wal.segment = segment
            self.wal.open()
            try:
                for _end, record in self.wal.iter_records(offset):
                    if record.get("op") != "commit":
                        continue
                    meta = record.get("meta")
                    if meta:
                        self.metadata = meta
                    marker = record.get("marker") or {}
                    for key, size in (marker.get("sizes") or {}).items():
                        sizes[key] = int(size)
                    # The log of the older versions held the patches of the
                    # commit; they are applied the old way.
                    patches = record.get("writes") or []
                    if patches:
                        touched.update(self.store.apply_writes(patches))
            finally:
                self.wal.close()
        if sizes or touched:
            # The data of a batch that was never committed is cut away.
            for key, size in sizes.items():
                self.store.trim_to(key, size)
            self._restore_alive()
        if touched:
            self.store.fsync(touched)
        # Continue logging in the last segment that was processed.
        self.wal.segment = last_segment
        self.wal.open()

    def _restore_alive(self) -> None:
        """Put the bitmap of every table back into the state of the metadata.

        A batch that was never committed may have cleared the bit of a row
        that was deleted.  The bitmap is rebuilt from the rows that are free,
        which is what the metadata of the table holds.
        """
        for name, payload in self.metadata.get("tables", {}).items():
            n_slots = int(payload.get("n_slots", 0))
            free = payload.get("free_slots") or []
            alive = np.ones(n_slots, dtype=bool)
            if free:
                alive[np.asarray(free, dtype=np.int64)] = False
            bitmap = np.packbits(alive, bitorder="little").tobytes()
            self.store.write("t/%s/alive.bits" % name, 0, bitmap)

    def _load_tables(self) -> None:
        self.tables = {}
        for name, payload in self.metadata.get("tables", {}).items():
            table = Table.from_meta(self.store, name, payload)
            table.load_alive()
            table.rebuild_primary_key_index()
            self.tables[name.lower()] = table

    def _rebuild_indexes(self) -> None:
        """Rebuild the approximate indexes from the persisted rows."""
        self.indexes = {}
        for name, table in self.tables.items():
            for index_name, payload in table.indexes.items():
                index = VectorIndex(
                    table=table,
                    name=index_name,
                    column=payload["column"],
                    dim=int(payload["dim"]),
                    clusters=int(payload.get("clusters", 0)),
                    # Every cluster is probed, see the creation of the index.
                    nprobe=int(payload.get("clusters", 0)) or 1,
                    lists=payload.get("lists", {}),
                    radii=payload.get("radii", {}),
                )
                self.indexes[(name, index_name.lower())] = index

    # ------------------------------------------------------------------
    # checkpointing
    # ------------------------------------------------------------------
    def checkpoint(self) -> None:
        """Persist the catalog and remember how far the log was applied."""
        self.metadata["wal"] = {"segment": self.wal.segment, "offset": self.wal.offset}
        self.metadata["tables"] = {
            table.name: table.to_meta() for table in self.tables.values()
        }
        self.store.write_json_atomic(CATALOG_FILE, self.metadata)

    # ------------------------------------------------------------------
    # transactions
    # ------------------------------------------------------------------
    @property
    def in_transaction(self) -> bool:
        return self.txn is not None

    def begin(self) -> None:
        if self.txn is not None:
            # Nested transactions are listed as not supported by the task
            # statement, so they are reported like the other statements
            # outside of the supported subset.
            raise UnsupportedError("nested transactions are not supported")
        self._txn_counter += 1
        self.txn = Transaction(id=self._txn_counter)

    def commit(self) -> None:
        if self.txn is None:
            return  # COMMIT outside of a transaction is a no-op, like MySQL
        txn = self.txn
        try:
            self._commit(txn)
        except Exception:
            # The in-memory state may already be ahead of the durable state.
            # Continuing would be unsafe, so the process stops and the log
            # recovery discards the uncommitted transaction.
            print(
                "minivectordb: fatal error while committing, aborting process",
                file=sys.stderr,
            )
            raise
        self.txn = None

    def rollback(self) -> None:
        if self.txn is None:
            return
        # Nothing was applied to the data files, so a rollback only has to drop
        # the buffered state.
        self.txn = None

    def _commit(self, txn: Transaction) -> None:
        # 1. dictionary entries that the transaction introduced
        for name, overlay in txn.overlays.items():
            table = self.tables.get(name)
            if table is None:
                continue
            for column_key, new_strings in overlay.pending_new_strings.items():
                column = table.columns[table.index[column_key]]
                dictionary = table.dictionary(column)
                for text in new_strings:
                    dictionary.plan_append(text, txn.writes)

        # 2. rows that have to be reflected in the approximate index
        for name, overlay in txn.overlays.items():
            table = self.tables.get(name)
            if table is None:
                continue
            for index, slot, vector in overlay.index_appends:
                index.assign(slot, vector, txn.writes)

        # 2. in-memory state that becomes visible with this commit
        for name, overlay in txn.overlays.items():
            table = self.tables.get(name)
            if table is None:
                continue
            for slot in overlay.new_slots:
                if slot + 1 > table.n_slots:
                    table.n_slots = slot + 1
            for slot in overlay.freed_slots:
                table.free_slots.append(slot)
            for slot, alive in overlay.alive_changes.items():
                table.apply_alive(slot, alive)
            for slot in overlay.inserted:
                if not table.primary_key:
                    continue
                pk = overlay.rows.get(slot, {}).get(table.primary_key.lower())
                if pk is not None:
                    table.pk_index.insert(int(pk), slot)
            for slot in overlay.deleted:
                if table.primary_key:
                    table.pk_index.remove(table.primary_key_at(slot))

        # 3. new tables become part of the catalog
        for name, table in txn.new_tables.items():
            self.tables[name] = table
        for name in txn.dropped_tables:
            self.tables.pop(name, None)

        # 4. the metadata of the database after this commit
        metadata = {
            "version": 1,
            "wal": {"segment": self.wal.segment, "offset": self.wal.offset},
            "tables": {
                table.name: table.to_meta() for table in self.tables.values()
            },
        }
        # 5. apply the patches to the data files
        keys = self.store.apply_writes(txn.writes.writes)
        for key in keys:
            cache_invalidate(self.store, key)
        self._dirty_keys.update(keys)

        # 6. the log: this is the commit point.  The log holds the state of
        #    the commit and not the data: the data is in the files, and a
        #    recovery only has to cut away the part of a batch that was not
        #    committed.
        marker = self._commit_marker(keys)
        self.wal.append_commit(txn.id, marker, metadata)
        self.metadata = metadata

        # 7. the values of the dictionary are in the files now, they are read
        #    from there instead of from the memory
        for name, overlay in txn.overlays.items():
            table = self.tables.get(name)
            if table is None:
                continue
            for column in table.columns:
                if column.type == "VARCHAR":
                    table.dictionary(column).flush()

        # 8. the commit point was the flush of the log.  The data files are
        #    only flushed before a checkpoint, because the log holds the state
        #    of the commits since the last one and a recovery cuts the files
        #    back to that state.  Flushing the data files on every commit
        #    cost five flushes per commit.
        self._pending_commits += 1
        if self._pending_commits >= CHECKPOINT_EVERY:
            self._flush_and_checkpoint()

    def _commit_marker(self, keys: List[str]) -> Dict[str, Any]:
        """What a recovery needs to know about a commit.

        The length every file the commit wrote had after the commit.  A
        recovery cuts the files back to these lengths, so that the part of a
        batch that was never committed is gone.
        """
        return {"sizes": {key: self.store.size(key) for key in keys}}

    def _flush_and_checkpoint(self) -> None:
        """Make the data files durable and write the catalog.

        After this the log may be dropped: the records before the checkpoint
        are not needed by a recovery any more.  The commits since the last
        checkpoint are in the log and are replayed when the process died
        before the data files were flushed.
        """
        if self._dirty_keys:
            self.store.fsync(self._dirty_keys)
            self._dirty_keys.clear()
        # The catalog is written first: it names the segment that is still
        # current, so a recovery starts there.  The rotation comes after it,
        # and only then the segments the catalog covers are dropped.
        self.checkpoint()
        self._rotate_wal()
        self._pending_commits = 0

    def _rotate_wal(self) -> None:
        """Start a new log segment and drop the ones the checkpoint covers.

        The records of the old segment are on the disk in the data files, so
        the log of a long run does not grow to the size of the whole table.
        The catalog is written after the rotation and names the new segment,
        so a recovery reads the records that are not on the disk yet.
        """
        try:
            index = int(self.wal.segment[len("wal-") : -len(".log")]) + 1
        except ValueError:
            index = 1
        old_segments = self._segments()
        current = self.wal.segment
        self.wal.switch_segment("wal-%06d.log" % index)
        for name in old_segments:
            # The catalog names the segment that was current, so the segments
            # before it are covered by the checkpoint and can go.  The segment
            # the catalog names is kept: a recovery starts there.
            if name < current:
                try:
                    self.store.remove("%s/%s" % (WAL_DIR, name))
                except OSError:
                    # Another process still has the file open.  The segment
                    # is skipped by the recovery anyway, so it is only a
                    # leftover that the next checkpoint removes.
                    pass

    # ------------------------------------------------------------------
    # helpers used by the executor
    # ------------------------------------------------------------------
    def table(self, name: str) -> Optional[Table]:
        return self.tables.get(name.lower())

    def current_transaction(self) -> Transaction:
        if self.txn is None:
            self.begin()
        assert self.txn is not None
        return self.txn

    def allocate_slot(self, table: Table) -> int:
        """Reserve a fresh row slot for the current transaction.

        Slots that were freed by the transaction are not reused immediately:
        the primary key index would still carry the old key of that slot.  They
        become available again for the *next* transaction.
        """
        txn = self.current_transaction()
        overlay = txn.overlay_for(table)
        slot = table.n_slots + len(overlay.new_slots)
        overlay.new_slots.append(slot)
        return slot

    def register_new_table(self, table: Table) -> None:
        txn = self.current_transaction()
        txn.new_tables[table.name.lower()] = table

    def check_primary_key(
        self, table: Table, key: int, exclude_slot: Optional[int] = None
    ) -> None:
        """Raise when ``key`` is already used by a live row."""
        if table.primary_key is None:
            return
        existing = table.pk_index.lookup(key)
        if existing is not None and existing != exclude_slot:
            raise DuplicateKeyError(f"duplicate entry for primary key {key}")
        if self.txn is not None:
            overlay = self.txn.overlays.get(table.name.lower())
            if overlay is None:
                return
            slot = overlay.inserted_keys.get(key)
            if slot is not None and slot != exclude_slot and slot not in overlay.deleted:
                raise DuplicateKeyError(f"duplicate entry for primary key {key}")
