"""The stdin/stdout protocol.

The organiser ships the SQL semantics with the task package but keeps the
process envelope in their own test driver.  This module therefore keeps the
envelope in one small, replaceable place:

* one statement per response, exactly one JSON object per statement on stdout,
  flushed immediately;
* diagnostics only on stderr.

The shape of the response objects follows the assertion format of the public
``mysqltest`` cases (2026-competition-round2):

===========================  ==========================================
statement                    response
===========================  ==========================================
``CREATE TABLE`` / DDL       ``{}``
``INSERT`` / ``UPDATE`` ...  ``{"affected_rows": n}``
``SELECT``                   ``{"columns": [...], "rows": [[...]]}``
failing statement            ``{"error": "duplicate_key"}``
===========================  ==========================================

The ``error`` value is the *error semantic name* of the task package.  The
numeric MySQL code, the SQLSTATE and the human readable message are written to
stderr so that stdout stays exactly what the driver compares.

Three rules keep the response stream in sync with the driver:

* **Statement boundaries.**  The input is read in chunks, not in lines, so a
  driver that sends ``SELECT 1;`` without a trailing newline works as well as
  one that sends whole lines.  A statement ends at the first top level ``;``.
  When the driver does not use ``;`` at all, the buffer is parsed instead: a
  statement that parses is executed, an incomplete one waits for more input.

* **Driver directives.**  ``--mode ann``, ``--restart`` and comments are *not*
  statements.  They are recognised, acted upon and answered with silence,
  because the task package states that a directive does not occupy an
  assertion position.

* **The readiness marker.**  A driver may either speak first or wait for a
  readiness marker.  Both are supported: the marker is written only when the
  process has been idle for a while, i.e. when the driver did not send
  anything yet.  A driver that speaks first therefore never sees an
  unexpected line on stdout.

Everything that is not part of that core can be switched with environment
variables, which keeps the program usable with different drivers:

``MINIVECTORDB_MODE``
    ``exact`` (default) or ``ann`` -- the retrieval mode.
``MINIVECTORDB_READY_MARKER``
    ``stdout`` (default), ``stderr``, ``both``, ``none`` or ``off``.
``MINIVECTORDB_READY_TEXT``
    The text of the ready marker.
``MINIVECTORDB_READY_DELAY_MS``
    How long the program waits for the first request before it announces that
    it is ready (default 5000 ms).
``MINIVECTORDB_STATEMENT_GRACE_MS``
    How long the program waits for more input when a statement without ``;``
    already parses (default 5 ms).  It only matters for drivers that split a
    statement over several writes.
"""

from __future__ import annotations

import codecs
import json
import os
import queue
import sys
import threading
from typing import Any, Dict, List, Optional, TextIO, Tuple

from .errors import MiniVectorDBError, ParseError
from .sqlparse import parse


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default
    return value if value >= 0 else default


#: How long the program waits for the first request before it announces that
#: it is ready.  A driver that speaks first sends its first statement long
#: before that, therefore it never sees the marker on stdout.
READY_DELAY_MS = _env_int("MINIVECTORDB_READY_DELAY_MS", 5000)

#: How long the program waits for more input when the buffered text already
#: parses as a complete statement but does not end with ``;``.
STATEMENT_GRACE_MS = _env_int("MINIVECTORDB_STATEMENT_GRACE_MS", 5)

READY_TEXT = os.environ.get("MINIVECTORDB_READY_TEXT", "READY")
READY_TARGET = os.environ.get("MINIVECTORDB_READY_MARKER", "stdout").strip().lower()

#: Directives that switch the retrieval mode of a running process.
MODE_DIRECTIVES = ("mode", "search-mode", "search_mode", "retrieval-mode", "retrieval_mode")

#: Directives that restart the process; the driver performs the restart.
RESTART_DIRECTIVES = ("restart", "reopen")


class StdinReader(threading.Thread):
    """Reads stdin in the background so that the server can use timeouts.

    The reader works on *chunks* instead of lines: a driver that sends a
    statement without a trailing newline would otherwise block the database
    until the pipe is closed.
    """

    def __init__(self, stream: TextIO):
        super().__init__(daemon=True)
        self.stream = stream
        self.queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self.eof = False
        self._peeked: Optional[str] = None
        self._peeked_valid = False
        self._first_chunk = True

    def _read_chunk(self) -> str:
        """Read the next piece of input, returning ``""`` at end of file."""
        raw = getattr(self.stream, "buffer", None)
        source = raw if raw is not None else self.stream
        try:
            if hasattr(source, "read1"):
                data = source.read1(4096)
            else:
                data = source.read(4096)
        except (OSError, ValueError):  # pragma: no cover - the pipe was closed
            return ""
        if not data:
            return ""
        if isinstance(data, bytes):
            return data.decode("utf-8", errors="replace")
        return data

    def run(self) -> None:
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        try:
            while True:
                chunk = self._read_chunk()
                if not chunk:
                    break
                if isinstance(chunk, bytes):  # pragma: no cover - defensive
                    chunk = decoder.decode(chunk)
                if self._first_chunk:
                    self._first_chunk = False
                    # A driver that pipes through a PowerShell command prepends
                    # a UTF-8 BOM.  It is an artefact of the transport, not part
                    # of the statement, so it is dropped instead of turning the
                    # first statement into a syntax error.
                    chunk = chunk.lstrip("\ufeff")
                    if not chunk:
                        continue
                self.queue.put(chunk)
        except Exception:  # pragma: no cover - the pipe was closed
            pass
        try:
            tail = decoder.decode(b"", True)
        except Exception:  # pragma: no cover - defensive
            tail = ""
        if tail:
            self.queue.put(tail)
        self.queue.put(None)

    def get(self, timeout: Optional[float] = None) -> Optional[str]:
        if self._peeked_valid:
            self._peeked_valid = False
            return self._peeked
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError

    def peek(self, timeout: Optional[float] = None) -> Optional[str]:
        """Return the next chunk without consuming it."""
        if not self._peeked_valid:
            try:
                self._peeked = self.queue.get(timeout=timeout)
            except queue.Empty:
                raise TimeoutError
            self._peeked_valid = True
        return self._peeked

    def pending(self) -> bool:
        """True when more input may already be available."""
        return self._peeked_valid or not self.queue.empty()


#: The scanner states of :func:`scan_statement_end`.
_CODE = ("code", "")
_LINE_COMMENT = ("line", "")
_BLOCK_COMMENT = ("block", "")


def scan_statement_end(
    text: str, index: int, state: Tuple[str, str]
) -> Tuple[int, Tuple[str, str]]:
    """Find the next top level ``;`` in ``text``, starting at ``index``.

    The scan is resumable: the caller keeps ``index`` and ``state`` and calls
    the function again with the next chunk of input.  A statement that arrives
    in many chunks is therefore scanned only once, which matters for the large
    multi row inserts of the performance benchmark.

    String literals and comments are skipped, so a semicolon inside a value
    does not split the statement.

    Returns ``(end, state)`` where ``end`` is the index just after the
    semicolon and ``-1`` when the text does not contain one (yet).
    """
    length = len(text)
    while index < length:
        kind = state[0]
        char = text[index]
        if kind == "code":
            if char == "'" or char == '"':
                state = ("string", char)
                index += 1
                continue
            if char == "`":
                state = ("ident", "")
                index += 1
                continue
            if char == "-" and text.startswith("--", index):
                state = _LINE_COMMENT
                index += 2
                continue
            if char == "#":
                state = _LINE_COMMENT
                index += 1
                continue
            if char == "/" and text.startswith("/*", index):
                state = _BLOCK_COMMENT
                index += 2
                continue
            if char == ";":
                return index + 1, _CODE
            index += 1
            continue
        if kind == "string":
            quote = state[1]
            if char == "\\":
                index += 2
                continue
            if char == quote:
                if text.startswith(quote * 2, index):
                    index += 2
                    continue
                state = _CODE
                index += 1
                continue
            index += 1
            continue
        if kind == "ident":
            if char == "`":
                if text.startswith("``", index):
                    index += 2
                    continue
                state = _CODE
                index += 1
                continue
            index += 1
            continue
        if kind == "line":
            if char == "\n":
                state = _CODE
            index += 1
            continue
        # a block comment
        if char == "*" and text.startswith("*/", index):
            state = _CODE
            index += 2
            continue
        index += 1
    return -1, state


def find_statement_end(text: str) -> int:
    """Return the index just after the first top level ``;``.

    String literals and comments are skipped, so a semicolon inside a value
    does not split the statement.
    """
    end, _state = scan_statement_end(text, 0, _CODE)
    if end < 0:
        return len(text)
    return end


def _parses(text: str) -> Tuple[bool, bool]:
    """Return ``(parses, incomplete)`` for a piece of text.

    ``parses`` is True when the text is a complete statement, ``incomplete``
    is True when the text ends in the middle of a statement, i.e. more input
    may complete it.
    """
    try:
        parse(text)
    except ParseError as error:
        return False, bool(getattr(error, "incomplete", False))
    except MiniVectorDBError:
        # A definite error (for example an unsupported construct): the text is
        # complete, it is simply not valid.
        return False, False
    return True, False


class Protocol:
    """Reads statements, executes them and writes the result objects."""

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
        #: The position of the resumable statement scan inside the buffer.
        self._scan_index = 0
        #: The state of the resumable statement scan at ``_scan_index``.
        self._scan_state = _CODE

    # -- output ---------------------------------------------------------
    def write_result(self, payload: Dict[str, Any]) -> None:
        self.stdout.write(json.dumps(payload, ensure_ascii=False, default=str))
        self.stdout.write("\n")
        self.stdout.flush()

    def announce_ready(self) -> None:
        target = self.ready_target
        if target in ("none", "off"):
            self.log("minivectordb: ready, waiting for statements")
            return
        if target in ("stdout", "both"):
            self.stdout.write(READY_TEXT + "\n")
            self.stdout.flush()
        self.stderr.write(READY_TEXT + "\n")
        self.stderr.flush()
        self.log("minivectordb: ready, waiting for statements")

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

    # -- main loop ------------------------------------------------------
    def run(self) -> int:
        reader = StdinReader(self.stdin)
        reader.start()
        buffer = ""
        announced = False
        eof = False
        while not eof:
            timeout = (
                READY_DELAY_MS / 1000.0
                if not buffer
                else max(STATEMENT_GRACE_MS, 1) / 1000.0
            )
            try:
                chunk = reader.get(timeout=timeout)
            except TimeoutError:
                chunk = ""
                idle = True
            else:
                idle = False
                if chunk is None:
                    eof = True
                    chunk = ""
            if not announced:
                if idle and not buffer:
                    # The driver did not send anything yet: it may be waiting
                    # for a readiness marker.
                    self.announce_ready()
                else:
                    # The driver speaks first, stdout carries results only.
                    self.log("minivectordb: ready, waiting for statements")
                announced = True
            buffer += chunk
            while True:
                statement, buffer = self._next_statement(buffer, reader, idle)
                if statement is None:
                    break
                self._handle(statement)
        # A trailing statement without a terminator.
        rest = buffer.strip()
        if rest:
            if rest.startswith("--"):
                self._handle_directive(rest)
            else:
                self._handle(rest)
        return 0

    def _next_statement(
        self, buffer: str, reader: "StdinReader", idle: bool
    ) -> Tuple[Optional[str], str]:
        """Return ``(statement, remainder)`` or ``(None, buffer)``.

        The rules are, in this order:

        1. a top level ``;`` ends the statement;
        2. a driver directive (a line that starts with ``--``) is consumed and
           handled separately, it never produces a response;
        3. if the whole buffer parses it is a complete statement (a driver that
           does not use ``;``); it is executed once the input went quiet;
        4. if the buffer does not parse but its first line does, the first line
           is the statement and the rest is kept for the next round.

        Deciding whether the buffer is a complete statement costs a full
        parse, therefore it is only done when the input went quiet.  A driver
        that streams a large statement into the pipe pays for one parse per
        statement instead of one per chunk.
        """
        # 1. the semicolon terminator
        end, state = scan_statement_end(buffer, self._scan_index, self._scan_state)
        if end >= 0:
            statement = buffer[:end]
            remainder = buffer[end:]
            self._scan_index = 0
            self._scan_state = _CODE
            if statement.strip():
                return statement, remainder
            return None, remainder
        # The whole buffer was scanned and holds no terminator (yet).
        self._scan_index = len(buffer)
        self._scan_state = state

        stripped = buffer.strip()
        if not stripped:
            # Only whitespace so far, drop it and wait for the statement.
            self._scan_index = 0
            self._scan_state = _CODE
            return None, ""

        # 2. driver directives
        if stripped.startswith("--"):
            newline = buffer.find("\n")
            if newline >= 0:
                line = buffer[:newline]
                remainder = buffer[newline + 1:]
                self._handle_directive(line.strip())
                self._scan_index = 0
                self._scan_state = _CODE
                return None, remainder
            # No newline yet: a known directive is complete, an unknown one may
            # still be a comment that continues in the next chunk.
            head = stripped[2:].strip().split(" ")[0].split("=")[0].lower()
            if head in MODE_DIRECTIVES or head in RESTART_DIRECTIVES or head in (
                "ann",
                "approximate",
                "exact",
                "brute-force",
                "brute_force",
            ):
                self._handle_directive(stripped)
                self._scan_index = 0
                self._scan_state = _CODE
                return None, ""
            return None, buffer

        # 3. the whole buffer is a statement
        if not idle:
            # More input may still complete the statement.  The completeness
            # check below costs a full parse, so it waits until the input
            # settles instead of running for every chunk.
            return None, buffer
        parses, incomplete = _parses(stripped)
        if parses:
            if self._continues(stripped):
                # The last line looks like the beginning of a clause, wait for
                # the rest of the statement.
                return None, buffer
            self._scan_index = 0
            self._scan_state = _CODE
            return stripped, ""
        if incomplete:
            # The statement is not finished yet, read more input.
            return None, buffer

        # 4. the buffer holds more than one statement
        newline = buffer.find("\n")
        if newline >= 0:
            first = buffer[:newline].strip()
            if first:
                first_parses, first_incomplete = _parses(first)
                if first_parses and not first_incomplete:
                    remainder = buffer[newline + 1:]
                    self._scan_index = 0
                    self._scan_state = _CODE
                    return first, remainder
        # A definite error: report it instead of waiting for more input.
        self._scan_index = 0
        self._scan_state = _CODE
        return stripped, ""

    #: Words that start a clause, i.e. a line that continues a statement.
    CONTINUATION_WORDS = (
        "where",
        "order",
        "limit",
        "offset",
        "group",
        "having",
        "and",
        "or",
        "set",
        "values",
        "join",
        "on",
        "from",
    )

    def _continues(self, text: str) -> bool:
        """True when the last line of the text continues a statement."""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) < 2:
            return False
        last = lines[-1]
        word = last.split(" ")[0].split("\t")[0].lower()
        if word in self.CONTINUATION_WORDS:
            return True
        return last.endswith((",", "(", "+", "-", "*", "/"))

    def _handle_directive(self, text: str) -> None:
        """Act on a driver directive without writing to stdout."""
        body = text[2:].strip()
        if not body:
            return
        parts = body.replace("=", " ").split()
        name = parts[0].lower()
        if name in MODE_DIRECTIVES:
            value = parts[1].lower() if len(parts) > 1 else ""
            if value in ("exact", "ann"):
                self._set_mode(value)
                return
            self.log(f"minivectordb: ignoring mode directive with value {value!r}")
            return
        if name in ("ann", "approximate"):
            self._set_mode("ann")
            return
        if name in ("exact", "brute-force", "brute_force"):
            self._set_mode("exact")
            return
        if name in RESTART_DIRECTIVES:
            # The driver restarts the process itself; nothing to do here.
            self.log("minivectordb: restart directive received")
            return
        self.log(f"minivectordb: ignoring unknown directive {text!r}")

    def _set_mode(self, mode: str) -> None:
        try:
            self.executor.db.mode = mode
        except AttributeError:  # pragma: no cover - defensive
            return
        self.log(f"minivectordb: retrieval mode is now {mode}")

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
            self.write_result({"error": "syntax_error"})
            return
        if not isinstance(request, dict):
            self.write_result({"error": "syntax_error"})
            return
        sql = request.get("sql") or request.get("statement") or request.get("query")
        if not isinstance(sql, str):
            self.write_result({"error": "syntax_error"})
            return
        mode = request.get("mode")
        if isinstance(mode, str) and mode.lower() in ("exact", "ann"):
            self._set_mode(mode.lower())
        payload = self._execute(sql)
        if "id" in request:
            payload = dict(payload)
            payload["id"] = request["id"]
        self.write_result(payload)

    def _execute(self, sql: str) -> Dict[str, Any]:
        try:
            statement = parse(sql)
        except ParseError as error:
            self.log_error(error)
            payload = error.to_dict()
            self.write_result(payload)
            return payload
        except MiniVectorDBError as error:
            self.log_error(error)
            payload = error.to_dict()
            self.write_result(payload)
            return payload
        try:
            payload = self.executor.execute(statement)
        except MiniVectorDBError as error:
            self.log_error(error)
            payload = error.to_dict()
            self.write_result(payload)
            return payload
        except Exception as error:  # pragma: no cover - defensive
            self.log(f"minivectordb: internal error: {error!r}")
            payload = {"error": "internal_error"}
            self.write_result(payload)
            return payload
        self.write_result(payload)
        return payload
