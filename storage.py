import json
import os
from contextlib import contextmanager
from pathlib import Path


def sync_directory(path):
    if os.name == "posix":
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def write_json(path, value):
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        stream.flush()
        os.fsync(stream.fileno())


def read_json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


class Storage:
    """A single writer stages changes on disk, then publishes them for redo."""

    def __init__(self, directory):
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=True)
        self.pending = self.root / "pending"
        self.committed = self.root / "committed"
        self.data = self.root / "rows"
        self.data.mkdir(exist_ok=True)
        self.catalog = self.root / "catalog.json"
        self.recover()
        self.rollback()

    def recover(self):
        if not self.committed.exists():
            return
        for staged_table in self.committed.iterdir():
            if not staged_table.is_dir():
                continue
            destination = self.data / staged_table.name
            destination.mkdir(exist_ok=True)
            with os.scandir(staged_table) as entries:
                for entry in entries:
                    source = Path(entry.path)
                    target = destination / entry.name
                    if source.stat().st_size == 0:
                        target.unlink(missing_ok=True)
                        sync_directory(destination)
                        source.unlink()
                    else:
                        os.replace(source, target)
            sync_directory(destination)
            staged_table.rmdir()
        sync_directory(self.data)
        catalog = self.committed / "catalog.json"
        if catalog.exists():
            os.replace(catalog, self.catalog)
        self.committed.rmdir()
        sync_directory(self.root)

    def rollback(self):
        if not self.pending.exists():
            return
        for item in self.pending.iterdir():
            if item.is_dir():
                with os.scandir(item) as entries:
                    for entry in entries:
                        os.unlink(entry.path)
                item.rmdir()
            else:
                item.unlink()
        self.pending.rmdir()

    def commit(self, catalog):
        self.pending.mkdir(exist_ok=True)
        write_json(self.pending / "catalog.json", catalog)
        for item in self.pending.iterdir():
            if item.is_dir():
                sync_directory(item)
        sync_directory(self.pending)
        os.rename(self.pending, self.committed)
        sync_directory(self.root)
        self.recover()

    @contextmanager
    def matched_keys(self, matches):
        # Freeze the statement's input before writing into its transaction overlay.
        self.pending.mkdir(exist_ok=True)
        path = self.pending / "matched-keys"
        try:
            with path.open("w+", encoding="ascii") as stream:
                for key, _ in matches:
                    stream.write(f"{key}\n")
                stream.seek(0)
                yield (int(line) for line in stream)
        finally:
            path.unlink(missing_ok=True)


class Rows:
    def __init__(self, storage, table):
        self.live = storage.data / ("table-" + table)
        self.staged = storage.pending / ("table-" + table)

    def __getitem__(self, key):
        path = self.staged / str(key)
        if path.exists():
            if path.stat().st_size == 0:
                raise KeyError(key)
            return read_json(path)
        try:
            return read_json(self.live / str(key))
        except FileNotFoundError as error:
            raise KeyError(key) from error

    def __contains__(self, key):
        path = self.staged / str(key)
        return path.stat().st_size > 0 if path.exists() else (self.live / str(key)).exists()

    def __setitem__(self, key, row):
        self.staged.mkdir(parents=True, exist_ok=True)
        write_json(self.staged / str(key), row)

    def __delitem__(self, key):
        self.staged.mkdir(parents=True, exist_ok=True)
        with (self.staged / str(key)).open("wb") as stream:
            os.fsync(stream.fileno())

    def items(self):
        if self.live.exists():
            with os.scandir(self.live) as entries:
                for entry in entries:
                    if not (self.staged / entry.name).exists():
                        yield int(entry.name), read_json(entry.path)
        if self.staged.exists():
            with os.scandir(self.staged) as entries:
                for entry in entries:
                    if entry.stat().st_size:
                        yield int(entry.name), read_json(entry.path)
