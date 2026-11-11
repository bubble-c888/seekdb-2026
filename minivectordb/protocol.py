"""The stdin/stdout protocol.

The organiser published the wire format together with the task package
(``STDIO_PROTOCOL.md`` and ``protocol_check/`` in the branch
``2026-competition-round2``).  It is a JSON Lines protocol:

* after the data directory has been opened and the recovery has finished, the
  program prints ``{"type":"ready"}`` and flushes it.  The marker is written
  before the first request is read, and every start writes it once;
* every request is one JSON object on one physical line, for example
  ``{"id":1,"op":"execute","sql":"SELECT id FROM docs;","search_mode":"exact"}``;
* every request is answered with exactly one JSON object that repeats the
  ``id`` unchanged (including its JSON type) and carries the boolean ``ok``;
* ``{"op":"shutdown"}`` rolls the open transaction back, persists everything,
  is answered with ``{"id":...,"ok":true}`` and ends the process with exit
  code 0.  Nothing may be written to stdout after that response;
* stdout carries protocol messages only, everything else goes to stderr.

The public ``mysqltest`` files are neither requests nor responses: a driver
wraps every statement of a ``.test`` file into an ``execute`` request and
compares the response against the assertions of the ``.result`` file.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, Iterator, Optional, TextIO, Tuple

from .errors import MiniVectorDBError
from .sqlparse import parse

#: The startup handshake of the task package.
READY_MESSAGE: Dict[str, str] = {"type": "ready"}

#: The operations the judge sends.
OP_EXECUTE = "execute"
OP_SHUTDOWN = "shutdown"

#: The retrieval modes the judge may ask for.
SEARCH_MODES = ("exact", "ann")


def _sanitize(value: Any) -> Any:
    """Replace the numbers JSON cannot represent by ``null``.

    The protocol forbids ``NaN`` and ``Infinity`` in the responses, and the
    judge parses the stream with ``allow_nan=False``.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    return value


class Protocol:
    """Reads requests from stdin and writes the answers to stdout."""

    def __init__(
        self,
        executor,
        stdin: TextIO,
        stdout: TextIO,
        stderr: TextIO,
        ready_target: Optional[str] = None,
    ):
        self.executor = executor
        self.stdin = stdin
        self.stdout = stdout
        self.stderr = stderr
        self.ready_target = (
            ready_target
            or os.environ.get("MINIVECTORDB_READY_MARKER", "stdout")
            or "stdout"
        ).strip().lower()

    # -- output ---------------------------------------------------------
    def write_message(self, payload: Dict[str, Any]) -> None:
        """Write one protocol message and flush it immediately."""
        data = json.dumps(
            _sanitize(payload), ensure_ascii=False, allow_nan=False, default=str
        )
        self.stdout.write(data)
        self.stdout.write("\n")
        self.stdout.flush()

    def announce_ready(self) -> None:
        """Write the startup handshake.

        The judge waits for exactly this object before it sends the first
        request, so it is written as soon as the database is open.
        """
        if self.ready_target in ("none", "off"):
            self.log("minivectordb: ready, waiting for requests")
            return
        self.write_message(dict(READY_MESSAGE))
        self.log("minivectordb: ready, waiting for requests")

    def log(self, message: str) -> None:
        self.stderr.write(message.rstrip("\n") + "\n")
        self.stderr.flush()

    def log_error(self, error: MiniVectorDBError) -> None:
        detail = error.to_log()
        self.log(
            "minivectordb: {error} (code {code}, sqlstate {sqlstate}): {message}".format(
                **detail
            )
        )

    # -- input ----------------------------------------------------------
    def _lines(self) -> Iterator[str]:
        """Yield the request lines of stdin.

        The requests are read from the byte stream when there is one, so that
        a request is never split by the decoding of the text layer.  A request
        is one physical line, the judge never sends more.
        """
        raw = getattr(self.stdin, "buffer", None)
        if raw is not None:
            for line in raw:
                yield line.decode("utf-8", errors="replace")
            return
        for line in self.stdin:
            yield line

    @staticmethod
    def _strip_bom(text: str) -> str:
        """Remove the byte order mark of the first line.

        A driver that pipes its input through a Windows shell prepends the
        mark.  It belongs to the transport, not to the request.
        """
        return text.lstrip("\ufeff")

    # -- main loop ------------------------------------------------------
    def run(self) -> int:
        """Serve the requests until the judge shuts the program down."""
        self.announce_ready()
        first = True
        for line in self._lines():
            text = line.strip()
            if first:
                first = False
                text = self._strip_bom(text)
            if not text:
                continue
            try:
                request = json.loads(text)
            except ValueError as error:
                self.log(f"minivectordb: ignoring a line that is not JSON: {error}")
                continue
            if not isinstance(request, dict):
                self.log("minivectordb: ignoring a request that is not a JSON object")
                continue
            if self.handle_request(request):
                return 0
        return 0

    def handle_request(self, request: Dict[str, Any]) -> bool:
        """Handle one request, return True when the program has to stop."""
        ident = request.get("id")
        operation = request.get("op")
        if operation == OP_SHUTDOWN:
            self.shutdown(ident)
            return True
        if operation != OP_EXECUTE:
            self.write_message(
                self.failure(ident, "unsupported_operation", f"unsupported operation {operation!r}")
            )
            return False
        sql = request.get("sql")
        if not isinstance(sql, str):
            self.write_message(
                self.failure(ident, "invalid_request", "the request carries no sql string")
            )
            return False
        self.execute(ident, sql, request.get("search_mode", "exact"))
        return False

    def execute(self, ident: Any, sql: str, mode: Any) -> None:
        """Run one statement and write its response."""
        if mode not in SEARCH_MODES:
            if mode is not None:
                self.log(f"minivectordb: unknown search_mode {mode!r}, using exact")
            mode = "exact"
        database = getattr(self.executor, "db", None)
        if database is not None:
            database.mode = mode
        try:
            result = self.executor.execute(parse(sql))
        except MiniVectorDBError as error:
            self.log_error(error)
            self.write_message(self.failure(ident, error.name, str(error)))
            return
        except Exception as error:  # pragma: no cover - defensive
            self.log(f"minivectordb: unexpected error: {error!r}")
            self.write_message(self.failure(ident, "execution_error", str(error)))
            return
        payload: Dict[str, Any] = {"id": ident, "ok": True}
        payload.update(result or {})
        self.write_message(payload)

    def shutdown(self, ident: Any) -> None:
        """Roll back, persist and answer the shutdown request."""
        database = getattr(self.executor, "db", None)
        try:
            if database is not None:
                if database.in_transaction:
                    database.rollback()
                database.close()
        except Exception as error:  # pragma: no cover - defensive
            self.log(f"minivectordb: error during shutdown: {error}")
        self.write_message({"id": ident, "ok": True})

    @staticmethod
    def failure(ident: Any, code: str, message: str) -> Dict[str, Any]:
        """The response of a failed statement."""
        return {
            "id": ident,
            "ok": False,
            "error": {"code": code or "execution_error", "message": message},
        }
