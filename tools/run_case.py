#!/usr/bin/env python3
"""Local self-test harness for the MiniVectorDB public mysqltest cases.

Every case in ``mysqltest/manifest.json`` is executed against a FRESH, EMPTY
data directory: the database program is started, fed the items of the ``.test``
file and every response line is compared with the matching entry of the
``.result`` array.

The ``.test`` files of the task package contain two kinds of items:

* a SQL statement -- it produces exactly one response line and therefore
  occupies exactly one position in the ``.result`` array;
* a driver directive (``--restart``, ``--mode ann``, comments) -- it does not
  produce a response and does not occupy an assertion position.

``--restart`` closes stdin, waits for the program to exit and starts it again
with the same data directory, exactly like the organiser describes it.

Usage examples::

    python tools/run_case.py                      # run every case
    python tools/run_case.py --case sql_crud      # run a single case
    python tools/run_case.py --data-dir .tmp/cases
    python tools/run_case.py --program "python -m minivectordb --data-dir <datadir>"
    python tools/run_case.py --keep --verbose

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
from typing import Any, List, Optional, Sequence, Tuple

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from compare import compare_result  # noqa: E402  (local module, path fixed above)

DEFAULT_PROGRAM = "{} -m minivectordb --data-dir <datadir>".format(sys.executable)
DEFAULT_TIMEOUT = 30.0
SHUTDOWN_TIMEOUT = 15.0
STDERR_TAIL_LINES = 40

#: Directives that make the harness restart the program.
RESTART_DIRECTIVES = ("restart", "reopen")


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def split_command(command: str) -> List[str]:
    """Split a command line into argv.

    ``shlex`` in POSIX mode eats Windows backslashes, so the non-POSIX lexer is
    used and matching quotes are stripped afterwards.
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


def build_command(program: str, data_dir: Path) -> List[str]:
    """Build the argv of the database program for one case."""
    tokens = split_command(program)
    if not tokens:
        raise ValueError("empty program command")
    return [token.replace("<datadir>", str(data_dir)) for token in tokens]


def format_command(tokens: Sequence[str]) -> str:
    """Render an argv as a copy/pasteable command line."""
    return " ".join(
        '"{}"'.format(token) if " " in token else token for token in tokens
    )


def format_value(value: Any) -> str:
    """Render a value as indented JSON."""
    try:
        return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False)
    except (TypeError, ValueError):
        return repr(value)


def stderr_tail(lines: Sequence[str], limit: int = STDERR_TAIL_LINES) -> str:
    """Return the tail of the captured stderr output."""
    tail = list(lines)[-limit:]
    return "".join(tail).rstrip()


# --------------------------------------------------------------------------
# case loading
# --------------------------------------------------------------------------
class Case:
    """One public mysqltest case."""

    def __init__(self, case_id: str, group: str, statements: int, test_file: Path, result_file: Path):
        self.case_id = case_id
        self.group = group
        self.declared_statements = statements
        self.test_file = test_file
        self.result_file = result_file


def load_cases(cases_dir: Path) -> List[Case]:
    """Load the case list from ``manifest.json``."""
    manifest_path = cases_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit("manifest not found: {}".format(manifest_path))
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    cases = []
    for entry in manifest.get("cases", []):
        case_id = entry["id"]
        cases.append(
            Case(
                case_id=case_id,
                group=entry.get("group", ""),
                statements=int(entry.get("statements", 0)),
                test_file=cases_dir / "cases" / "{}.test".format(case_id),
                result_file=cases_dir / "cases" / "{}.result".format(case_id),
            )
        )
    return cases


def read_statements(test_file: Path) -> List[str]:
    """Read the statements of a ``.test`` file: one item per non-empty line."""
    with test_file.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle.read().splitlines() if line.strip()]


def read_items(test_file: Path) -> List[Tuple[str, str]]:
    """Read a ``.test`` file as a list of ``(kind, text)`` items.

    ``kind`` is ``"sql"`` for a statement and ``"directive"`` for a driver
    instruction such as ``--restart`` or ``--mode ann``.  Only the statements
    occupy a position in the ``.result`` file.
    """
    items: List[Tuple[str, str]] = []
    for line in read_statements(test_file):
        if line.startswith("--"):
            items.append(("directive", line))
        else:
            items.append(("sql", line))
    return items


def read_expectations(result_file: Path) -> List[dict]:
    """Read the expected assertion objects of a ``.result`` file."""
    with result_file.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise SystemExit("result file is not a JSON array: {}".format(result_file))
    return payload


# --------------------------------------------------------------------------
# running one case
# --------------------------------------------------------------------------
class Failure:
    """One recorded failure inside a case."""

    def __init__(self, message: str, statement_index: Optional[int] = None,
                 sql: Optional[str] = None, expected: Any = None,
                 actual: Any = None, problems: Optional[Sequence[str]] = None):
        self.message = message
        self.statement_index = statement_index
        self.sql = sql
        self.expected = expected
        self.actual = actual
        self.problems = list(problems or [])


class CaseResult:
    """Outcome of one case."""

    def __init__(self, case_id: str):
        self.case_id = case_id
        self.passed = False
        self.total = 0
        self.checked = 0
        self.failures: List[Failure] = []
        self.duration = 0.0
        self.stderr = ""


def _pump_stdout(stream, out_queue: "queue.Queue", errors: List[str]) -> None:
    """Forward stdout lines to a queue; push ``None`` on EOF."""
    try:
        for line in stream:
            if "\ufffd" in line and not errors:
                errors.append(
                    "the response stream contains bytes that are not valid UTF-8. "
                    "The database program must write UTF-8 encoded JSON on stdout."
                )
            out_queue.put(line)
    except Exception as exc:  # pragma: no cover - defensive
        errors.append("cannot read the response stream: {}".format(exc))
    finally:
        out_queue.put(None)


def _pump_stderr(stream, sink: List[str]) -> None:
    """Drain stderr continuously so the child never blocks on a full pipe."""
    try:
        for line in stream:
            sink.append(line)
    except Exception:  # pragma: no cover - defensive
        pass


class Program:
    """One run of the database program together with its pipes."""

    def __init__(self, command: Sequence[str], environment: dict, cwd: str):
        self.command = list(command)
        self.environment = environment
        self.cwd = cwd
        self.process: Optional[subprocess.Popen] = None
        self.stdout_queue: "queue.Queue" = queue.Queue()
        self.stderr_lines: List[str] = []
        self.stdout_errors: List[str] = []
        self._stdout_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None

    # -- lifecycle ------------------------------------------------------
    def start(self) -> None:
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            cwd=self.cwd,
            env=self.environment,
        )
        self.stdout_queue = queue.Queue()
        self.stderr_lines = []
        self.stdout_errors = []
        self._stdout_thread = threading.Thread(
            target=_pump_stdout,
            args=(self.process.stdout, self.stdout_queue, self.stdout_errors),
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=_pump_stderr, args=(self.process.stderr, self.stderr_lines), daemon=True
        )
        self._stdout_thread.start()
        self._stderr_thread.start()

    def send(self, text: str) -> None:
        assert self.process is not None
        self.process.stdin.write(text + "\n")
        self.process.stdin.flush()

    def read_line(self, timeout: float) -> Optional[str]:
        """Return the next response line, ``None`` on EOF and ``TIMEOUT`` on timeout."""
        try:
            return self.stdout_queue.get(timeout=timeout)
        except queue.Empty:
            return TIMEOUT

    def close_stdin(self) -> None:
        if self.process is None or self.process.stdin is None:
            return
        try:
            if not self.process.stdin.closed:
                self.process.stdin.close()
        except Exception:  # pragma: no cover - defensive
            pass

    def wait(self, timeout: float) -> Optional[int]:
        if self.process is None:
            return None
        try:
            return self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None

    def kill(self) -> None:
        if self.process is None:
            return
        try:
            self.process.kill()
        except Exception:  # pragma: no cover - defensive
            pass

    def close_pipes(self) -> None:
        if self.process is None:
            return
        for stream in (self.process.stdout, self.process.stderr):
            try:
                if stream is not None:
                    stream.close()
            except Exception:  # pragma: no cover - defensive
                pass
        for thread in (self._stdout_thread, self._stderr_thread):
            if thread is not None:
                thread.join(timeout=2.0)

    def drain_extra_lines(self) -> List[str]:
        """Return the response lines that no statement claimed."""
        extra: List[str] = []
        while True:
            try:
                line = self.stdout_queue.get_nowait()
            except queue.Empty:
                break
            if line is None:
                continue
            if line.strip():
                extra.append(line.strip())
        return extra


class _Timeout:
    """Marker for a read that timed out."""

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return "<timeout>"


TIMEOUT = _Timeout()


def run_case(case: Case, program: str, parent_dir: Path, timeout: float,
             verbose: bool) -> CaseResult:
    """Run one case against a fresh database and compare every response."""
    result = CaseResult(case.case_id)
    started = time.monotonic()

    items = read_items(case.test_file)
    expectations = read_expectations(case.result_file)
    result.total = sum(1 for kind, _ in items if kind == "sql")

    data_dir = parent_dir / case.case_id
    if data_dir.exists():
        shutil.rmtree(str(data_dir), ignore_errors=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    try:
        command = build_command(program, data_dir)
    except ValueError as exc:
        result.failures.append(Failure(str(exc)))
        result.duration = time.monotonic() - started
        return result

    if verbose:
        print("+ {}".format(format_command(command)), flush=True)

    environment = os.environ.copy()
    # The response protocol is UTF-8; make sure the child does not fall back
    # to the console code page of the current locale (Windows).
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"

    runner = Program(command, environment, str(_REPO_ROOT))
    try:
        runner.start()
    except OSError as exc:
        result.failures.append(
            Failure("cannot start the database program: {}".format(exc))
        )
        result.duration = time.monotonic() - started
        return result

    statement_index = 0
    stopped = False

    try:
        for kind, text in items:
            if kind == "directive":
                name = text[2:].strip().split(" ")[0].split("=")[0].lower()
                if name in RESTART_DIRECTIVES:
                    # Normal shutdown, wait for the exit, start again with the
                    # same data directory.
                    runner.close_stdin()
                    code = runner.wait(SHUTDOWN_TIMEOUT)
                    if code is None:
                        runner.kill()
                        result.failures.append(
                            Failure(
                                "the database program did not exit within {:.1f}s "
                                "of the restart directive".format(SHUTDOWN_TIMEOUT)
                            )
                        )
                        stopped = True
                        break
                    if code != 0:
                        result.failures.append(
                            Failure(
                                "the database program exited with code {} at the "
                                "restart directive".format(code)
                            )
                        )
                    runner.close_pipes()
                    runner = Program(command, environment, str(_REPO_ROOT))
                    try:
                        runner.start()
                    except OSError as exc:
                        result.failures.append(
                            Failure("cannot restart the database program: {}".format(exc))
                        )
                        stopped = True
                        break
                    if verbose:
                        print("--- restart", flush=True)
                    continue
                # Every other directive is forwarded to the program; it does
                # not answer a directive.
                try:
                    runner.send(text)
                except (BrokenPipeError, OSError) as exc:
                    result.failures.append(
                        Failure(
                            "the database program stopped accepting input at the "
                            "directive {!r}: {}".format(text, exc)
                        )
                    )
                    stopped = True
                    break
                continue

            expectation = (
                expectations[statement_index]
                if statement_index < len(expectations)
                else None
            )
            try:
                runner.send(text)
            except (BrokenPipeError, OSError) as exc:
                exit_code = runner.process.poll()
                result.failures.append(
                    Failure(
                        "the database program stopped accepting input after {} statement(s) "
                        "(exit code {}): {}".format(statement_index, exit_code, exc),
                        statement_index=statement_index,
                        sql=text,
                    )
                )
                stopped = True
                break

            line = runner.read_line(timeout)
            if line is TIMEOUT:
                runner.kill()
                result.failures.append(
                    Failure(
                        "timeout after {:.1f}s waiting for the response to statement #{}".format(
                            timeout, statement_index + 1
                        ),
                        statement_index=statement_index,
                        sql=text,
                    )
                )
                stopped = True
                break

            if line is None:
                exit_code = runner.process.poll()
                if exit_code is None:
                    exit_code = runner.wait(SHUTDOWN_TIMEOUT)
                result.failures.append(
                    Failure(
                        "the database program exited (exit code {}) without answering "
                        "statement #{}".format(exit_code, statement_index + 1),
                        statement_index=statement_index,
                        sql=text,
                    )
                )
                stopped = True
                break

            stripped = line.strip()
            if not stripped:
                result.failures.append(
                    Failure(
                        "the database program produced an empty response line for statement #{}".format(
                            statement_index + 1
                        ),
                        statement_index=statement_index,
                        sql=text,
                    )
                )
                statement_index += 1
                continue

            try:
                actual = json.loads(stripped)
            except ValueError as exc:
                result.failures.append(
                    Failure(
                        "the response to statement #{} is not valid JSON ({}): {}".format(
                            statement_index + 1, exc, stripped
                        ),
                        statement_index=statement_index,
                        sql=text,
                        actual=stripped,
                    )
                )
                statement_index += 1
                continue

            if expectation is None:
                result.failures.append(
                    Failure(
                        "the result file has no expectation for statement #{}".format(
                            statement_index + 1
                        ),
                        statement_index=statement_index,
                        sql=text,
                        actual=actual,
                    )
                )
                statement_index += 1
                continue

            problems = compare_result(expectation, actual)
            result.checked += 1
            if verbose:
                print("--- statement #{}: {}".format(statement_index + 1, text), flush=True)
                print("expected: {}".format(format_value(expectation)), flush=True)
                print("actual:   {}".format(format_value(actual)), flush=True)
            if problems:
                result.failures.append(
                    Failure(
                        "statement #{} does not match the expectation".format(
                            statement_index + 1
                        ),
                        statement_index=statement_index,
                        sql=text,
                        expected=expectation,
                        actual=actual,
                        problems=problems,
                    )
                )
            statement_index += 1

        # Every statement must have produced exactly one response line.
        if statement_index > len(expectations):
            for index in range(len(expectations), statement_index):
                result.failures.append(
                    Failure(
                        "the result file has no expectation for statement #{}".format(index + 1),
                        statement_index=index,
                    )
                )

        for error in runner.stdout_errors:
            result.failures.append(Failure(error))

    finally:
        if not stopped:
            runner.close_stdin()
            exit_code = runner.wait(SHUTDOWN_TIMEOUT)
            if exit_code is None:
                runner.kill()
                try:
                    runner.wait(SHUTDOWN_TIMEOUT)
                except Exception:  # pragma: no cover - defensive
                    pass
                result.failures.append(
                    Failure(
                        "the database program did not exit within {:.1f}s after stdin was closed".format(
                            SHUTDOWN_TIMEOUT
                        )
                    )
                )
            elif exit_code != 0:
                result.failures.append(
                    Failure("the database program exited with code {}".format(exit_code))
                )
        else:
            runner.kill()
            runner.wait(SHUTDOWN_TIMEOUT)

        runner.close_pipes()

        # Any response line that was not claimed by a statement is a protocol
        # violation (the program must answer exactly one line per statement).
        for extra in runner.drain_extra_lines():
            result.failures.append(
                Failure("unexpected extra output line: {}".format(extra))
            )

    result.stderr = stderr_tail(runner.stderr_lines)
    result.duration = time.monotonic() - started
    result.passed = not result.failures
    return result


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def print_failures(result: CaseResult) -> None:
    """Print the machine readable failure detail of one case."""
    for failure in result.failures:
        location = ""
        if failure.statement_index is not None:
            location = "statement #{} (index {})".format(
                failure.statement_index + 1, failure.statement_index
            )
        print("  {}: {}".format(location or "case", failure.message), flush=True)
        if failure.sql is not None:
            print("    sql: {}".format(failure.sql), flush=True)
        for problem in failure.problems:
            print("    - {}".format(problem), flush=True)
        if failure.expected is not None:
            print("    expected:", flush=True)
            print(format_value(failure.expected), flush=True)
        if failure.actual is not None:
            print("    actual:", flush=True)
            print(format_value(failure.actual), flush=True)
    if result.stderr:
        print("  stderr (tail):", flush=True)
        for line in result.stderr.splitlines():
            print("    {}".format(line), flush=True)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_case.py",
        description=(
            "Run the public MiniVectorDB mysqltest cases against the database "
            "program. Every case runs on a fresh, empty data directory."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python tools/run_case.py\n"
            "  python tools/run_case.py --case sql_crud\n"
            "  python tools/run_case.py --data-dir .tmp/cases --keep\n"
            "  python tools/run_case.py --program \"python -m minivectordb --data-dir <datadir>\"\n"
        ),
    )
    parser.add_argument(
        "--case",
        action="append",
        metavar="NAME",
        help="run only the given case (may be repeated); default: every case in mysqltest/manifest.json",
    )
    parser.add_argument(
        "--data-dir",
        metavar="DIR",
        help=(
            "parent directory for the per-case data directories; "
            "default: a temporary directory that is removed afterwards"
        ),
    )
    parser.add_argument(
        "--program",
        default=DEFAULT_PROGRAM,
        metavar="CMD",
        help=(
            "command that starts the database program; the placeholder <datadir> is "
            "replaced by the per-case data directory; "
            "default: \"{}\"".format(DEFAULT_PROGRAM)
        ),
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="keep the per-case data directories instead of deleting them",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print every statement together with the expected and actual objects",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        metavar="SECONDS",
        help="timeout in seconds for a single statement (default: %(default)s)",
    )
    parser.add_argument(
        "--cases-dir",
        metavar="DIR",
        help="directory holding manifest.json and the cases/ sub directory (default: mysqltest)",
    )
    return parser


def _configure_output_encoding() -> None:
    """Emit UTF-8 when the output is piped.

    On Windows a redirected stdout uses the console code page, which would make
    the SQL statements and the failure details unreadable for the caller.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            if not stream.isatty():
                stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # pragma: no cover - defensive
            pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    _configure_output_encoding()
    parser = build_parser()
    args = parser.parse_args(argv)

    cases_dir = Path(args.cases_dir).resolve() if args.cases_dir else (_REPO_ROOT / "mysqltest")
    try:
        all_cases = load_cases(cases_dir)
    except (OSError, ValueError, KeyError) as exc:
        print("error: cannot load the cases: {}".format(exc), file=sys.stderr)
        return 2

    if args.case:
        wanted = list(dict.fromkeys(args.case))
        known = {case.case_id for case in all_cases}
        unknown = [name for name in wanted if name not in known]
        if unknown:
            print(
                "error: unknown case(s): {} (known: {})".format(
                    ", ".join(unknown), ", ".join(sorted(known))
                ),
                file=sys.stderr,
            )
            return 2
        selected = [case for case in all_cases if case.case_id in wanted]
    else:
        selected = list(all_cases)

    if not selected:
        print("error: no case selected", file=sys.stderr)
        return 2

    temporary_parent: Optional[Path] = None
    if args.data_dir:
        parent_dir = Path(args.data_dir).resolve()
        parent_dir.mkdir(parents=True, exist_ok=True)
    else:
        temporary_parent = Path(tempfile.mkdtemp(prefix="minivectordb-selftest-"))
        parent_dir = temporary_parent

    results: List[CaseResult] = []
    try:
        for case in selected:
            result = run_case(
                case,
                program=args.program,
                parent_dir=parent_dir,
                timeout=args.timeout,
                verbose=args.verbose,
            )
            results.append(result)
            if result.passed:
                print(
                    "[PASS] {} statements={}/{} duration={:.3f}s".format(
                        result.case_id, result.checked, result.total, result.duration
                    ),
                    flush=True,
                )
            else:
                print(
                    "[FAIL] {} statements={}/{} duration={:.3f}s failures={}".format(
                        result.case_id,
                        result.checked,
                        result.total,
                        result.duration,
                        len(result.failures),
                    ),
                    flush=True,
                )
                print_failures(result)
            if not args.keep:
                shutil.rmtree(str(parent_dir / case.case_id), ignore_errors=True)
    finally:
        if temporary_parent is not None and not args.keep:
            shutil.rmtree(str(temporary_parent), ignore_errors=True)

    passed = sum(1 for result in results if result.passed)
    total = len(results)
    print(
        "summary: {}/{} cases passed ({})".format(
            passed,
            total,
            ", ".join(result.case_id for result in results if not result.passed) or "no failures",
        ),
        flush=True,
    )
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
