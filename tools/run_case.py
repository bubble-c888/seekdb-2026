#!/usr/bin/env python3
"""Local self-test harness for the public mysqltest cases.

The harness speaks the JSON Lines protocol of the task package (see
``STDIO_PROTOCOL.md``) and compares the responses with the assertions of the
``.result`` files **using the assertion code of the task package**
(``protocol_check/_assertions.py``), so that the local verdict and the verdict
of the judge use the same rules.

Every case runs against a fresh, empty data directory:

* the program is started and the ``{"type":"ready"}`` handshake is read;
* every statement of the ``.test`` file becomes one ``execute`` request;
* ``--restart`` shuts the program down, waits for the exit and starts it again
  with the same data directory;
* ``--mode ann`` / ``--mode exact`` change the ``search_mode`` of the
  following requests;
* the program is shut down at the end, the exit code and the rest of stdout
  are checked.

Usage examples::

    python tools/run_case.py                      # run every case
    python tools/run_case.py --case sql_crud      # run a single case
    python tools/run_case.py --program "python -m minivectordb --data-dir <datadir>"
    python tools/run_case.py --verbose

Exit status is 0 only when every selected case passes.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# The judge compares the answers on Linux; the local harness runs on Windows as
# well, where the console code page would mangle the Chinese messages.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):  # pragma: no cover - defensive
        pass

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
sys.path.insert(0, str(_REPO_ROOT / "protocol_check"))

from _assertions import assert_response  # noqa: E402  (task package module)

DEFAULT_PROGRAM = "{} -m minivectordb --data-dir <datadir>".format(sys.executable)
DEFAULT_TIMEOUT = 60.0
READY_MESSAGE = {"type": "ready"}


def split_command(command: str) -> List[str]:
    """Split a command line into argv.

    ``shlex`` in POSIX mode eats Windows backslashes, so the non-POSIX lexer
    is used and matching quotes are stripped afterwards.
    """
    lexer = shlex.shlex(command, posix=False)
    lexer.whitespace_split = True
    lexer.commenters = ""
    tokens: List[str] = []
    for token in lexer:
        if len(token) >= 2 and token[0] == '"' and token[-1] == '"':
            token = token[1:-1]
        tokens.append(token)
    return tokens


class Program:
    """The database program under test.

    stdout and stderr are drained by two threads, so that a program that
    writes a lot of diagnostics cannot block, and so that the harness can
    apply its own timeout to every single answer.
    """

    def __init__(self, argv: Sequence[str], cwd: Optional[Path] = None):
        self.argv = list(argv)
        self.cwd = str(cwd) if cwd else None
        self.process: Optional[subprocess.Popen] = None
        self.stderr_lines: List[str] = []
        self.sequence = 0
        self.timeout = DEFAULT_TIMEOUT
        self._lines: "queue.Queue[Optional[bytes]]" = queue.Queue()
        self._reader: Optional[threading.Thread] = None
        self._error_reader: Optional[threading.Thread] = None

    # -- process --------------------------------------------------------
    def start(self) -> None:
        environment = dict(os.environ)
        # The protocol is UTF-8; the console code page of a Windows host must
        # not decide how the answers are encoded.
        environment["PYTHONIOENCODING"] = "utf-8"
        environment.setdefault("PYTHONUNBUFFERED", "1")
        self.process = subprocess.Popen(
            self.argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.cwd,
            bufsize=0,
            env=environment,
        )
        self._lines = queue.Queue()
        self.stderr_lines = []
        self._reader = threading.Thread(target=self._drain_stdout, daemon=True)
        self._reader.start()
        self._error_reader = threading.Thread(target=self._drain_stderr, daemon=True)
        self._error_reader.start()

    def _drain_stdout(self) -> None:
        stream = self.process.stdout if self.process else None
        if stream is None:  # pragma: no cover - defensive
            return
        try:
            for line in stream:
                self._lines.put(line)
        except Exception:  # pragma: no cover - defensive
            pass
        self._lines.put(None)

    def _drain_stderr(self) -> None:
        stream = self.process.stderr if self.process else None
        if stream is None:  # pragma: no cover - defensive
            return
        try:
            for line in stream:
                self.stderr_lines.append(line.decode("utf-8", "replace").rstrip("\n"))
        except Exception:  # pragma: no cover - defensive
            pass

    # -- messages -------------------------------------------------------
    def read_line(self, timeout: Optional[float] = None) -> str:
        """Wait for one line of stdout."""
        limit = self.timeout if timeout is None else timeout
        try:
            line = self._lines.get(timeout=limit)
        except queue.Empty:
            raise TimeoutError(
                "timeout after {:.1f}s while waiting for an answer".format(limit)
            )
        if line is None:
            raise RuntimeError(
                "the program exited before the answer; stderr:\n" + self.stderr_tail()
            )
        return line.decode("utf-8")

    def send(self, request: Dict[str, Any]) -> None:
        assert self.process is not None and self.process.stdin is not None
        data = json.dumps(request, ensure_ascii=False, allow_nan=False) + "\n"
        self.process.stdin.write(data.encode("utf-8"))
        self.process.stdin.flush()

    def request(
        self, sql: Optional[str], mode: str = "exact", op: str = "execute"
    ) -> Dict[str, Any]:
        self.sequence += 1
        payload: Dict[str, Any] = {"id": self.sequence, "op": op}
        if sql is not None:
            payload["sql"] = sql
            payload["search_mode"] = mode
        self.send(payload)
        response = json.loads(self.read_line())
        if response.get("id") != payload["id"]:
            raise RuntimeError("the response id does not match the request")
        if not isinstance(response.get("ok"), bool):
            raise RuntimeError("the response has no boolean ok")
        return response

    def shutdown(self) -> Dict[str, Any]:
        response = self.request(None, op="shutdown")
        assert self.process is not None
        self.process.stdin.close()
        code = self.process.wait(timeout=self.timeout)
        # Nothing may follow the shutdown answer on stdout.
        extra = []
        while True:
            try:
                line = self._lines.get_nowait()
            except queue.Empty:
                break
            if line:
                extra.append(line.decode("utf-8", "replace").rstrip("\n"))
        if extra:
            raise RuntimeError(
                "the program wrote to stdout after the shutdown: " + " | ".join(extra)
            )
        if code != 0:
            raise RuntimeError("the program exited with code {}".format(code))
        return response

    def stderr_tail(self, limit: int = 40) -> str:
        return "\n".join(self.stderr_lines[-limit:])


def read_items(test_file: Path) -> List[Tuple[str, str]]:
    """Read a ``.test`` file as a list of ``(kind, text)`` items."""
    items: List[Tuple[str, str]] = []
    for line in test_file.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        items.append(("directive" if text.startswith("--") else "sql", text))
    return items


def run_case(
    case_id: str,
    package: Path,
    argv: Sequence[str],
    verbose: bool,
    timeout: float = DEFAULT_TIMEOUT,
) -> Tuple[bool, str]:
    """Run one case and return ``(passed, message)``."""
    test_file = package / "cases" / "{}.test".format(case_id)
    result_file = package / "cases" / "{}.result".format(case_id)
    expectations = json.loads(result_file.read_text(encoding="utf-8"))
    data_dir = Path(tempfile.mkdtemp(prefix="minivectordb-case-"))
    program = Program([token.replace("<datadir>", str(data_dir)) for token in argv])
    program.timeout = timeout
    index = 0
    mode = "exact"
    try:
        program.start()
        ready = json.loads(program.read_line())
        if ready != READY_MESSAGE:
            return False, "expected the ready handshake, got {!r}".format(ready)
        for kind, text in read_items(test_file):
            if kind == "directive":
                if text == "--restart":
                    program.shutdown()
                    program.start()
                    ready = json.loads(program.read_line())
                    if ready != READY_MESSAGE:
                        return False, "the restart did not send the handshake"
                elif text in ("--mode exact", "--mode ann"):
                    mode = text.split()[1]
                else:
                    return False, "unsupported directive {}".format(text)
                continue
            if index >= len(expectations):
                return False, "the case has more statements than assertions"
            response = program.request(text, mode)
            if verbose:
                print("--- statement #{}: {}".format(index + 1, text))
                print("--- answer #{}: {}".format(index + 1, json.dumps(response, ensure_ascii=False)))
            try:
                assert_response(
                    response, expectations[index], {"abs_tol": 1e-5, "rel_tol": 1e-5}
                )
            except AssertionError as error:
                return False, "statement {}: {}".format(index + 1, error)
            index += 1
        if index != len(expectations):
            return False, "the case has more assertions than statements"
        program.shutdown()
        return True, "{} statements".format(index)
    except Exception as error:  # noqa: BLE001
        return False, "{}: {}".format(type(error).__name__, error)
    finally:
        if program.process is not None and program.process.poll() is None:
            program.process.kill()
        shutil.rmtree(str(data_dir), ignore_errors=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_case.py",
        description="Run the public mysqltest cases against the local program.",
    )
    parser.add_argument("--case", action="append", help="public case id")
    parser.add_argument(
        "--cases-dir",
        default=str(_REPO_ROOT / "mysqltest"),
        help="directory of the task package (default: mysqltest)",
    )
    parser.add_argument(
        "--program",
        default=DEFAULT_PROGRAM,
        help="the program under test; <datadir> is replaced by the data directory",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="seconds the program may take for one answer (default: 60)",
    )
    parser.add_argument("--verbose", action="store_true", help="print every case")
    arguments = parser.parse_args(argv)

    package = Path(arguments.cases_dir)
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    cases = [entry["id"] for entry in manifest["cases"]]
    if arguments.case:
        unknown = set(arguments.case) - set(cases)
        if unknown:
            parser.error("unknown public case: " + ", ".join(sorted(unknown)))
        cases = [case for case in cases if case in arguments.case]

    argv_program = split_command(arguments.program)
    passed = 0
    failures = []
    for case_id in cases:
        started = time.perf_counter()
        ok, message = run_case(
            case_id, package, argv_program, arguments.verbose, arguments.timeout
        )
        duration = time.perf_counter() - started
        if ok:
            passed += 1
            print(
                "[PASS] {} statements={} duration={:.3f}s".format(
                    case_id, message, duration
                )
            )
        else:
            failures.append((case_id, message))
            print("[FAIL] {}: {}".format(case_id, message))
    print(
        "summary: {}/{} cases passed{}".format(
            passed,
            len(cases),
            "" if not failures else " (failures)",
        )
    )
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
