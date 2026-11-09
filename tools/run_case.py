#!/usr/bin/env python3
"""Local self-test harness for the MiniVectorDB public mysqltest cases.

Every case in ``mysqltest/manifest.json`` is executed against a FRESH, EMPTY
data directory: the database program is started, fed the statements of the
``.test`` file one per line on stdin, and every response line is compared with
the matching entry of the ``.result`` array.

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
from typing import Any, List, Optional, Sequence

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from compare import compare_result  # noqa: E402  (local module, path fixed above)

DEFAULT_PROGRAM = "{} -m minivectordb --data-dir <datadir>".format(sys.executable)
DEFAULT_TIMEOUT = 30.0
SHUTDOWN_TIMEOUT = 15.0
STDERR_TAIL_LINES = 40


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
    """Read the statements of a ``.test`` file: one statement per non-empty line."""
    with test_file.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle.read().splitlines() if line.strip()]


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
            if "�" in line and not errors:
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


def run_case(case: Case, program: str, parent_dir: Path, timeout: float,
             verbose: bool) -> CaseResult:
    """Run one case against a fresh database and compare every response."""
    result = CaseResult(case.case_id)
    started = time.monotonic()

    statements = read_statements(case.test_file)
    expectations = read_expectations(case.result_file)
    result.total = len(statements)

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

    try:
        environment = os.environ.copy()
        # The response protocol is UTF-8; make sure the child does not fall back
        # to the console code page of the current locale (Windows).
        environment["PYTHONIOENCODING"] = "utf-8"
        environment["PYTHONUTF8"] = "1"
        environment["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            cwd=str(_REPO_ROOT),
            env=environment,
        )
    except OSError as exc:
        result.failures.append(
            Failure("cannot start the database program: {}".format(exc))
        )
        result.duration = time.monotonic() - started
        return result

    stdout_queue: "queue.Queue" = queue.Queue()
    stderr_lines: List[str] = []
    stdout_errors: List[str] = []
    stdout_thread = threading.Thread(
        target=_pump_stdout,
        args=(process.stdout, stdout_queue, stdout_errors),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_pump_stderr, args=(process.stderr, stderr_lines), daemon=True
    )
    stdout_thread.start()
    stderr_thread.start()

    killed = False

    def kill_process() -> None:
        nonlocal killed
        if killed:
            return
        killed = True
        try:
            process.kill()
        except Exception:  # pragma: no cover - defensive
            pass

    try:
        for index, sql in enumerate(statements):
            expectation = expectations[index] if index < len(expectations) else None
            try:
                process.stdin.write(sql + "\n")
                process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                exit_code = process.poll()
                result.failures.append(
                    Failure(
                        "the database program stopped accepting input after {} statement(s) "
                        "(exit code {}): {}".format(index, exit_code, exc),
                        statement_index=index,
                        sql=sql,
                    )
                )
                break

            try:
                line = stdout_queue.get(timeout=timeout)
            except queue.Empty:
                kill_process()
                result.failures.append(
                    Failure(
                        "timeout after {:.1f}s waiting for the response to statement #{}".format(
                            timeout, index + 1
                        ),
                        statement_index=index,
                        sql=sql,
                    )
                )
                break

            if line is None:
                exit_code = process.poll()
                if exit_code is None:
                    try:
                        exit_code = process.wait(timeout=SHUTDOWN_TIMEOUT)
                    except subprocess.TimeoutExpired:
                        kill_process()
                        exit_code = "unknown (killed)"
                result.failures.append(
                    Failure(
                        "the database program exited (exit code {}) without answering "
                        "statement #{}".format(exit_code, index + 1),
                        statement_index=index,
                        sql=sql,
                    )
                )
                break

            text = line.strip()
            if not text:
                result.failures.append(
                    Failure(
                        "the database program produced an empty response line for statement #{}".format(
                            index + 1
                        ),
                        statement_index=index,
                        sql=sql,
                    )
                )
                continue

            try:
                actual = json.loads(text)
            except ValueError as exc:
                result.failures.append(
                    Failure(
                        "the response to statement #{} is not valid JSON ({}): {}".format(
                            index + 1, exc, text
                        ),
                        statement_index=index,
                        sql=sql,
                        actual=text,
                    )
                )
                continue

            if expectation is None:
                result.failures.append(
                    Failure(
                        "the result file has no expectation for statement #{}".format(index + 1),
                        statement_index=index,
                        sql=sql,
                        actual=actual,
                    )
                )
                continue

            problems = compare_result(expectation, actual)
            result.checked += 1
            if verbose:
                print("--- statement #{}: {}".format(index + 1, sql), flush=True)
                print("expected: {}".format(format_value(expectation)), flush=True)
                print("actual:   {}".format(format_value(actual)), flush=True)
            if problems:
                result.failures.append(
                    Failure(
                        "statement #{} does not match the expectation".format(index + 1),
                        statement_index=index,
                        sql=sql,
                        expected=expectation,
                        actual=actual,
                        problems=problems,
                    )
                )

        # Every statement must have produced exactly one response line.
        if len(statements) > len(expectations):
            for index in range(len(expectations), len(statements)):
                result.failures.append(
                    Failure(
                        "the result file has no expectation for statement #{}".format(index + 1),
                        statement_index=index,
                        sql=statements[index],
                    )
                )

        # Look for responses that were never claimed by a statement.  The check
        # runs again after the program exited, because the response lines can
        # only be complete once stdout has been closed.
        for error in stdout_errors:
            result.failures.append(Failure(error))

    finally:
        try:
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
        except Exception:  # pragma: no cover - defensive
            pass
        try:
            exit_code = process.wait(timeout=SHUTDOWN_TIMEOUT)
        except subprocess.TimeoutExpired:
            kill_process()
            try:
                process.wait(timeout=SHUTDOWN_TIMEOUT)
            except subprocess.TimeoutExpired:  # pragma: no cover - defensive
                exit_code = None
            result.failures.append(
                Failure(
                    "the database program did not exit within {:.1f}s after stdin was closed".format(
                        SHUTDOWN_TIMEOUT
                    )
                )
            )
        else:
            if exit_code != 0:
                result.failures.append(
                    Failure("the database program exited with code {}".format(exit_code))
                )
        try:
            process.stdout.close()
            process.stderr.close()
        except Exception:  # pragma: no cover - defensive
            pass
        stderr_thread.join(timeout=2.0)
        stdout_thread.join(timeout=2.0)

        # Any response line that was not claimed by a statement is a protocol
        # violation (the program must answer exactly one line per statement).
        while True:
            try:
                extra = stdout_queue.get_nowait()
            except queue.Empty:
                break
            if extra is not None and extra.strip():
                result.failures.append(
                    Failure("unexpected extra output line: {}".format(extra.strip()))
                )
                break

    result.stderr = stderr_tail(stderr_lines)
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
