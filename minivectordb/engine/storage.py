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
import zlib
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

#: The magic at the beginning of a log file that uses the binary format.
#: An older log is one JSON object per line; the reader recognises it by the
#: missing magic and still understands it.
WAL_MAGIC = b"MVDBWAL1"

#: The fixed fields of the binary format.  A record is the length and the
#: checksum of the payload, the payload itself, and the payload is the
#: transaction, the metadata snapshot and the physical byte patches.
_WAL_HEADER = struct.Struct("<II")
_WAL_RECORD = struct.Struct("<QII")
_WAL_PATCH = struct.Struct("<HQI")


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

    def set_size(self, key: str, size: int) -> None:
        """Remember the size of a file that was changed outside of a write."""
        self._sizes[key] = size

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
                # The patches that follow each other without a gap are
                # written together: the rows of a batch are appended one
                # after the other, so a hundred patches become one write.
                index = 0
                while index < len(patches):
                    start, data = patches[index]
                    index += 1
                    while index < len(patches) and patches[index][0] == start + len(data):
                        data = data + patches[index][1]
                        index += 1
                    handle.seek(start)
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

    One record per committed transaction.  The record holds the physical
    patches that have to be re-applied during a recovery and the metadata
    snapshot that becomes valid with the commit.

    The record is written in a binary format: a length, a checksum, the
    metadata and the patches as they are.  The patches of a large batch are
    mostly vector data, and the text format had to escape and to encode them,
    which made the log of a batch bigger than the data it protects and cost
    more time than writing the data itself.

    A log that an older version wrote (one JSON object per line) is still
    read: the reader recognises the format by the magic at the beginning of
    the file.
    """

    def __init__(self, store: FileStore, directory: str = "wal"):
        self.store = store
        self.directory = directory
        self.segment = "wal-000001.log"
        self.offset = 0
        self._handle = None
        self._binary = True

    # -- lifecycle ------------------------------------------------------
    @property
    def key(self) -> str:
        return f"{self.directory}/{self.segment}"

    def open(self) -> None:
        path = self.store.path(self.key)
        path.parent.mkdir(parents=True, exist_ok=True)
        size = path.stat().st_size if path.exists() else 0
        self._handle = open(path, "r+b" if path.exists() else "w+b")
        if size == 0:
            # A new file starts with the magic of the binary format.
            self._binary = True
            self._handle.write(WAL_MAGIC)
            self._handle.flush()
        else:
            self._handle.seek(0)
            self._binary = self._handle.read(len(WAL_MAGIC)) == WAL_MAGIC
        self._handle.seek(0, os.SEEK_END)
        self.offset = self._handle.tell()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()
            self._handle = None

    def switch_segment(self, segment: str, offset: int = 0) -> None:
        """Continue in another segment; the writes go to the end of it."""
        self.close()
        self.segment = segment
        self.offset = 0
        self.open()

    # -- writing --------------------------------------------------------
    def append_commit(self, txn_id: int, writes: WriteSet, meta: Dict[str, Any]) -> int:
        """Append one committed transaction and flush it to stable storage."""
        assert self._handle is not None, "log is not open"
        meta_bytes = json.dumps(meta, ensure_ascii=False, sort_keys=True).encode("utf-8")
        patches = bytearray()
        count = 0
        for key, offset, data in writes.writes:
            key_bytes = key.encode("utf-8")
            patches += _WAL_PATCH.pack(len(key_bytes), offset, len(data))
            patches += key_bytes
            patches += data
            count += 1
        payload = _WAL_RECORD.pack(txn_id, len(meta_bytes), count) + meta_bytes + bytes(patches)
        record = _WAL_HEADER.pack(len(payload), zlib.crc32(payload) & 0xFFFFFFFF) + payload
        self._handle.seek(self.offset)
        self._handle.write(record)
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self.offset += len(record)
        return self.offset

    def trim(self, size: int) -> None:
        """Cut the log at ``size``.

        The records before ``size`` are on the disk in the data files, so a
        recovery does not need them any more.  The file is cut at ``size``
        and the writes continue there.
        """
        if self._handle is None:
            return
        path = self.store.path(self.key)
        if not path.exists():
            return
        # The size of the file itself, not the size the store remembers: the
        # store does not see the writes of the log and would report an old
        # one, and the log would then be cut at the wrong place.
        current = path.stat().st_size
        if current <= size:
            return
        self._handle.flush()
        os.ftruncate(self._handle.fileno(), size)
        self._handle.seek(size)
        self.offset = size
        self.store.set_size(self.key, size)

    # -- reading --------------------------------------------------------
    def iter_records(self, start: int = 0) -> Iterator[Tuple[int, Dict[str, Any]]]:
        """Yield ``(offset_after_record, record)`` for every complete record.

        A record that a crash cut in half is ignored, and the file is cut at
        the end of the last complete record.
        """
        path = self.store.path(self.key)
        if not path.exists():
            return
        if self._binary:
            yield from self._iter_binary(path, start)
        else:
            yield from self._iter_lines(path, start)

    def _iter_binary(self, path: Path, start: int) -> Iterator[Tuple[int, Dict[str, Any]]]:
        offset = start
        good_end = start
        with open(path, "rb", buffering=1 << 20) as handle:
            handle.seek(0)
            if handle.read(len(WAL_MAGIC)) != WAL_MAGIC:
                return
            if start < len(WAL_MAGIC):
                # The records begin behind the magic.
                start = len(WAL_MAGIC)
                offset = start
                good_end = start
            handle.seek(start)
            while True:
                header = handle.read(_WAL_HEADER.size)
                if len(header) < _WAL_HEADER.size:
                    break
                length, checksum = _WAL_HEADER.unpack(header)
                payload = handle.read(length)
                if len(payload) < length:
                    break  # the record was cut in half
                if zlib.crc32(payload) & 0xFFFFFFFF != checksum:
                    break  # the record was cut in half
                record = self._decode(payload)
                if record is None:
                    break
                offset += _WAL_HEADER.size + length
                good_end = offset
                yield offset, record
        self._cut(path, good_end)

    def _iter_lines(self, path: Path, start: int) -> Iterator[Tuple[int, Dict[str, Any]]]:
        """Read a log of the older format: one JSON object per line."""
        good_end = start
        with open(path, "rb") as handle:
            handle.seek(start)
            offset = start
            for raw in handle:
                if not raw.endswith(b"\n"):
                    break  # torn write at the end of the log
                text = raw.strip()
                offset += len(raw)
                if not text:
                    good_end = offset
                    continue
                try:
                    record = json.loads(text.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    good_end = offset - len(raw)
                    break
                record["writes"] = WriteSet.decode(record.get("writes", []))
                good_end = offset
                yield offset, record
        self._cut(path, good_end)

    @staticmethod
    def _decode(payload: bytes) -> Optional[Dict[str, Any]]:
        """Turn the payload of a record back into its parts."""
        try:
            txn_id, meta_length, count = _WAL_RECORD.unpack_from(payload, 0)
            position = _WAL_RECORD.size
            meta = json.loads(payload[position : position + meta_length].decode("utf-8"))
            position += meta_length
            writes: List[Tuple[str, int, bytes]] = []
            for _ in range(count):
                key_length, offset, length = _WAL_PATCH.unpack_from(payload, position)
                position += _WAL_PATCH.size
                key = payload[position : position + key_length].decode("utf-8")
                position += key_length
                writes.append((key, offset, payload[position : position + length]))
                position += length
        except (struct.error, UnicodeDecodeError, json.JSONDecodeError, IndexError):
            return None
        if position != len(payload):
            return None
        return {"op": "commit", "txn": txn_id, "writes": writes, "meta": meta}

    def _cut(self, path: Path, good_end: int) -> None:
        """Cut the log at the end of the last complete record."""
        size = path.stat().st_size
        if good_end >= size:
            # The reader started behind the end of the file: there is no torn
            # tail, and the file must not grow.
            if good_end > size:
                self.offset = size
            return
        if self._handle is not None and path == self.store.path(self.key):
            self._handle.flush()
            os.ftruncate(self._handle.fileno(), good_end)
            self._handle.seek(good_end)
        else:
            with open(path, "r+b") as handle:
                os.ftruncate(handle.fileno(), good_end)
        self.store.set_size(self.key, good_end)
        self.offset = good_end


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
