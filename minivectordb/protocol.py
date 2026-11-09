"""The stdin/stdout protocol.

The organiser ships the SQL semantics with the task package but keeps the
process envelope in their own test driver.  This module therefore keeps the
envelope in one small, replaceable place:

* one statement per input line (a statement may span several lines);
* exactly one JSON object per statement on stdout, flushed immediately;
* diagnostics only on stderr.

The shape of the response objects follows the assertion format of the public
``mysqltest`` cases: ``{}`` for DDL, ``{"affected_rows": n}`` for DML,
``{"columns": [...], "rows": [[...]]}`` for queries and
``{"error": ..., "code": ..., "sqlstate": ...}`` for failures.

Everything that is not part of that core can be switched with environment
variables, which keeps the program usable with different drivers:

``MINIVECTORDB_MODE``
    ``exact`` (default) or ``ann`` -- the retrieval mode.
``MINIVECTORDB_READY_MARKER``
    ``stdout`` (default), ``stderr``, ``none`` or ``off``.
``MINIVECTORDB_READY_TEXT``
    The text of the ready marker.
``MINIVECTORDB_READY_DELAY_MS``
    How long the program waits for the first request before it announces that
    it is ready (default 400 ms).  A driver that speaks first therefore never
    sees an unexpected line on stdout.
"""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
from typing import Any, Dict, Optional, TextIO

from .errors import MiniVectorDBError, ParseError
from .sqlparse import parse

READY_DELAY_MS = int(os.environ.get("MINIVECTORDB_READY_DELAY_MS", "400") or 400)
READY_TEXT = os.environ.get("MINIVECTORDB_READY_TEXT", "READY")
READY_TARGET = os.environ.get("MINIVECTORDB_READY_MARKER", "stdout").strip().lower()


class StdinReader(threading.Thread):
    """Reads stdin in the background so that the server can use timeouts."""

    def __init__(self, stream: TextIO):
        super().__init__(daemon=True)
        self.stream = stream
        self.queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self.eof = False

    def run(self) -> None:
        try:
            for line in self.stream:
                self.queue.put(line)
        except Exception:  # pragma: no cover - the pipe was closed
            pass
        self.queue.put(None)

    def get(self, timeout: Optional[float] = None) -> Optional[str]:
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError


def find_statement_end(text: str) -> int:
    """Return the index just after the first top level ``;``.

    String literals and comments are skipped, so a semicolon inside a value
    does not split the statement.
    """
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char == "'":
            index += 1
            while index < length:
                if text[index] == "\\":
                    index += 2
                    continue
                if text[index] == "'":
                    if index + 1 < length and text[index + 1] == "'":
                        index += 2
                        continue
                    index += 1
                    break
                index += 1
            continue
        if char == "-" and text.startswith("--", index):
            end = text.find("\n", index)
            index = length if end == -1 else end + 1
            continue
        if char == "#":
            end = text.find("\n", index)
            index = length if end == -1 else end + 1
            continue
        if char == "/" and text.startswith("/*", index):
            end = text.find("*/", index + 2)
            index = length if end == -1 else end + 2
            continue
        if char == ";":
            return index + 1
        index += 1
    return length


class Protocol:
    """Reads statements, executes them and writes the result objects."""

    def __init__(self, executor, stdin: TextIO, stdout: TextIO, stderr: TextIO):
        self.executor = executor
        self.stdin = stdin
        self.stdout = stdout
        self.stderr = stderr

    # -- output ---------------------------------------------------------
    def write_result(self, payload: Dict[str, Any]) -> None:
        self.stdout.write(json.dumps(payload, ensure_ascii=False, default=str))
        self.stdout.write("\n")
        self.stdout.flush()

    def announce_ready(self) -> None:
        if READY_TARGET in ("none", "off"):
            return
        if READY_TARGET in ("stdout", "both"):
            self.stdout.write(READY_TEXT + "\n")
            self.stdout.flush()
        if READY_TARGET in ("stderr", "both"):
            self.stderr.write(READY_TEXT + "\n")
            self.stderr.flush()

    def log(self, message: str) -> None:
        self.stderr.write(message.rstrip("\n") + "\n")
        self.stderr.flush()

    # -- main loop ------------------------------------------------------
    def run(self) -> int:
        reader = StdinReader(self.stdin)
        reader.start()
        buffer = ""
        announced = False
        while True:
            try:
                line = reader.get(timeout=None if buffer else READY_DELAY_MS / 1000.0)
            except TimeoutError:
                if not announced:
                    self.announce_ready()
                    announced = True
                continue
            if line is None:
                break
            buffer += line
            if not announced and buffer.strip():
                announced = True
            while True:
                statement_text, buffer = self._split_statement(buffer)
                if statement_text is None:
                    break
                self._handle(statement_text)
        # a trailing statement without newline
        if buffer.strip():
            self._handle(buffer)
        return 0

    def _split_statement(self, buffer: str):
        """Return ``(statement, remainder)`` or ``(None, buffer)``."""
        stripped = buffer.strip()
        if not stripped:
            return None, ""
        end = find_statement_end(buffer)
        statement = buffer[:end]
        remainder = buffer[end:]
        if not statement.strip():
            return None, remainder
        return statement, remainder

    def _handle(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        if text.startswith("{"):
            self._handle_json_request(text)
            return
        self._execute(text)

    def _handle_json_request(self, text: str) -> None:
        try:
            request = json.loads(text)
        except json.JSONDecodeError:
            self.write_result(
                {
                    "error": "invalid request envelope",
                    "code": 1064,
                    "sqlstate": "42000",
                }
            )
            return
        if not isinstance(request, dict):
            self.write_result(
                {"error": "invalid request envelope", "code": 1064, "sqlstate": "42000"}
            )
            return
        sql = request.get("sql") or request.get("statement") or request.get("query")
        if not isinstance(sql, str):
            self.write_result(
                {"error": "request does not contain a statement", "code": 1064, "sqlstate": "42000"}
            )
            return
        mode = request.get("mode")
        if isinstance(mode, str) and mode.lower() in ("exact", "ann"):
            self.executor.db.mode = mode.lower()
        payload = self._execute(sql)
        if "id" in request:
            payload = dict(payload)
            payload["id"] = request["id"]
        self.write_result(payload)

    def _execute(self, sql: str) -> Dict[str, Any]:
        try:
            statement = parse(sql)
        except ParseError as error:
            payload = {
                "error": error.message,
                "code": error.code,
                "sqlstate": error.sqlstate,
            }
            self.write_result(payload)
            return payload
        except MiniVectorDBError as error:
            payload = error.to_dict()
            self.write_result(payload)
            return payload
        try:
            payload = self.executor.execute(statement)
        except MiniVectorDBError as error:
            payload = error.to_dict()
            self.write_result(payload)
            return payload
        except Exception as error:  # pragma: no cover - defensive
            self.log(f"minivectordb: internal error: {error!r}")
            payload = {
                "error": f"internal error: {error}",
                "code": 1105,
                "sqlstate": "HY000",
            }
            self.write_result(payload)
            return payload
        self.write_result(payload)
        return payload
