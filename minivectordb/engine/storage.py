"""Physical storage layer of MiniVectorDB.

The database is a *column store* on top of plain files:

* every table owns one fixed width file per column (``t/<table>/col_<name>.dat``),
  where slot ``i`` of the file holds the value of row ``i``;
* ``t/<table>/alive.bits`` is a bitmap of live rows;
* ``t/<table>/dict_<name>.dat`` / ``.idx`` hold the string dictionary of a
  ``VARCHAR`` column (dictionary encoding keeps the column fixed width while
  still allowing the required Unicode code point ordering);
* writes go through a write ahead log so that a committed transaction survives
  a ``SIGKILL`` and an uncommitted transaction never becomes partially visible.

Nothing in this module uses an off-the-shelf storage engine: the file format,
the log format and the recovery procedure are implemented here.
"""

from __future__ import annotations

import base64
import json
import os
import struct
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

# A WAL record is one JSON object per line.  ``writes`` holds the physical
# byte patches that have to be re-applied during recovery.
_WAL_HEADER = b""


class FileStore:
    """Thin, crash safe wrapper around the files below the data directory."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._sizes: Dict[str, int] = {}
        self._paths: Dict[str, Path] = {}

    # -- path handling -------------------------------------------------
    def path(self, key: str) -> Path:
        cached = self._paths.get(key)
        if cached is None:
            cached = self.root / key
            self._paths[key] = cached
        return cached

    def _ensure_parent(self, key: str) -> Path:
        path = self.path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    # -- reading --------------------------------------------------------
    def size(self, key: str) -> int:
        cached = self._sizes.get(key)
        if cached is not None:
            return cached
        path = self.path(key)
        size = path.stat().st_size if path.exists() else 0
        self._sizes[key] = size
        return size

    def read(self, key: str, offset: int = 0, length: Optional[int] = None) -> bytes:
        path = self.path(key)
        try:
            with open(path, "rb") as handle:
                handle.seek(offset)
                return handle.read() if length is None else handle.read(length)
        except FileNotFoundError:
            return b""

    def read_all(self, key: str) -> bytes:
        return self.read(key, 0, None)

    # -- writing --------------------------------------------------------
    def write(self, key: str, offset: int, data: bytes) -> None:
        """Write ``data`` at ``offset`` and flush it to the operating system."""
        if not data:
            return
        path = self._ensure_parent(key)
        with open(path, "r+b" if path.exists() else "w+b") as handle:
            handle.seek(offset)
            handle.write(data)
            handle.flush()
        self._sizes[key] = max(self.size(key), offset + len(data))

    def append(self, key: str, data: bytes) -> int:
        """Append ``data`` and return the offset it was written at."""
        if not data:
            return self.size(key)
        path = self._ensure_parent(key)
        offset = self.size(key)
        with open(path, "r+b" if path.exists() else "w+b") as handle:
            handle.seek(offset)
            handle.write(data)
            handle.flush()
        self._sizes[key] = offset + len(data)
        return offset

    def truncate(self, key: str, size: int) -> None:
        path = self._ensure_parent(key)
        with open(path, "r+b" if path.exists() else "w+b") as handle:
            handle.truncate(size)
            handle.flush()
        self._sizes[key] = size

    def remove(self, key: str) -> None:
        path = self.path(key)
        if path.exists():
            path.unlink()
        self._sizes.pop(key, None)

    def fsync(self, keys: Iterable[str]) -> None:
        """Force the given files (and the data directory) to stable storage."""
        for key in keys:
            path = self.path(key)
            if not path.exists():
                continue
            # A writable handle is required for fsync on Windows.
            with open(path, "r+b") as handle:
                os.fsync(handle.fileno())
        self.fsync_dir(self.root)

    def apply_writes(self, writes: Iterable[Tuple[str, int, bytes]]) -> List[str]:
        """Apply byte patches, one open per file, and return the touched keys."""
        grouped: Dict[str, List[Tuple[int, bytes]]] = {}
        for key, offset, data in writes:
            grouped.setdefault(key, []).append((offset, data))
        for key, patches in grouped.items():
            path = self._ensure_parent(key)
            with open(path, "r+b" if path.exists() else "w+b") as handle:
                for offset, data in patches:
                    handle.seek(offset)
                    handle.write(data)
                handle.flush()
            end = max(offset + len(data) for offset, data in patches)
            self._sizes[key] = max(self.size(key), end)
        return list(grouped)

    def fsync_dir(self, directory: Path) -> None:
        try:
            fd = os.open(str(directory), os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(fd)
        except OSError:
            pass
        finally:
            os.close(fd)

    # -- catalog --------------------------------------------------------
    def write_json_atomic(self, key: str, payload: Dict[str, Any]) -> None:
        path = self._ensure_parent(key)
        tmp = path.with_suffix(path.suffix + ".tmp")
        data = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        with open(tmp, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        self.fsync_dir(path.parent)
        self._sizes[key] = len(data)

    def read_json(self, key: str) -> Optional[Dict[str, Any]]:
        path = self.path(key)
        if not path.exists():
            return None
        try:
            with open(path, "rb") as handle:
                return json.loads(handle.read().decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None


class WriteSet:
    """Physical byte patches that a transaction wants to apply.

    The patches are only handed to :class:`FileStore` when the transaction is
    committed, therefore an aborted transaction never touches the data files.
    Besides the patches the set remembers the logical end of every file it
    touches, so that two appends to the same file inside one transaction do
    not overwrite each other.
    """

    def __init__(self) -> None:
        self._writes: List[Tuple[str, int, bytes]] = []
        self._sizes: Dict[str, int] = {}

    def add(self, key: str, offset: int, data: bytes) -> None:
        if not data:
            return
        self._writes.append((key, offset, data))
        end = offset + len(data)
        if end > self._sizes.get(key, 0):
            self._sizes[key] = end

    def logical_size(self, key: str, physical_size: int) -> int:
        """The end offset of ``key`` including the patches of this transaction."""
        return max(physical_size, self._sizes.get(key, 0))

    def extend(self, writes: Iterable[Tuple[str, int, bytes]]) -> None:
        for key, offset, data in writes:
            self.add(key, offset, data)

    @property
    def writes(self) -> List[Tuple[str, int, bytes]]:
        return self._writes

    def touched_keys(self) -> List[str]:
        seen: List[str] = []
        for key, _, _ in self._writes:
            if key not in seen:
                seen.append(key)
        return seen

    def __len__(self) -> int:
        return len(self._writes)

    def encode(self) -> List[List[Any]]:
        return [
            [key, offset, base64.b64encode(data).decode("ascii")]
            for key, offset, data in self._writes
        ]

    @staticmethod
    def decode(entries: Iterable[Iterable[Any]]) -> List[Tuple[str, int, bytes]]:
        decoded: List[Tuple[str, int, bytes]] = []
        for entry in entries:
            key, offset, payload = entry
            decoded.append((key, int(offset), base64.b64decode(payload)))
        return decoded


class WriteAheadLog:
    """Append only log of committed transactions.

    One JSON object per line, each line describing a single committed
    transaction: the physical patches it produced and the metadata snapshot
    that becomes valid once the transaction is committed.
    """

    def __init__(self, store: FileStore, directory: str = "wal"):
        self.store = store
        self.directory = directory
        self.segment = "wal-000001.log"
        self.offset = 0
        self._handle = None

    # -- lifecycle ------------------------------------------------------
    @property
    def key(self) -> str:
        return f"{self.directory}/{self.segment}"

    def open(self) -> None:
        path = self.store.path(self.key)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(path, "r+b" if path.exists() else "w+b")
        self._handle.seek(0, os.SEEK_END)
        self.offset = self._handle.tell()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()
            self._handle = None

    def switch_segment(self, segment: str, offset: int = 0) -> None:
        self.close()
        self.segment = segment
        self.offset = offset
        self.open()

    # -- writing --------------------------------------------------------
    def append_commit(self, txn_id: int, writes: WriteSet, meta: Dict[str, Any]) -> int:
        """Append one committed transaction and flush it to stable storage."""
        assert self._handle is not None, "log is not open"
        record = {
            "op": "commit",
            "txn": txn_id,
            "writes": writes.encode(),
            "meta": meta,
        }
        payload = json.dumps(record, ensure_ascii=False).encode("utf-8") + b"\n"
        self._handle.seek(self.offset)
        self._handle.write(payload)
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self.offset += len(payload)
        return self.offset

    # -- reading --------------------------------------------------------
    def iter_records(self, start: int = 0) -> Iterator[Tuple[int, Dict[str, Any]]]:
        """Yield ``(offset_after_record, record)`` for every complete record.

        A trailing partial line (a transaction that was cut in half by a crash)
        is ignored, and the file is truncated to the last complete record.
        """
        path = self.store.path(self.key)
        if not path.exists():
            return
        good_end = 0
        with open(path, "rb") as handle:
            handle.seek(start)
            offset = start
            for raw in handle:
                if not raw.endswith(b"\n"):
                    break  # torn write at the end of the log
                text = raw.strip()
                if not text:
                    offset += len(raw)
                    good_end = offset
                    continue
                try:
                    record = json.loads(text.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    break
                offset += len(raw)
                good_end = offset
                yield offset, record
        if good_end != self.store.size(self.key):
            self.store.truncate(self.key, good_end)


# ---------------------------------------------------------------------------
# fixed width helpers
# ---------------------------------------------------------------------------


def pack_int64(value: int) -> bytes:
    return struct.pack("<q", int(value))


def unpack_int64(data: bytes) -> int:
    return struct.unpack("<q", data[:8])[0]


def pack_int32(value: int) -> bytes:
    return struct.pack("<i", int(value))


def pack_float32_array(values: Iterable[float]) -> bytes:
    import numpy as np

    return np.asarray(list(values), dtype=np.float32).tobytes()
