#!/usr/bin/env python3
"""End-to-end smoke test of the mysqltest harness.

The test drives ``tools/run_case.py`` against a tiny fake database program that
is written to a temporary file on the fly.  The fake only implements the
protocol of the task package (the ``{"type":"ready"}`` handshake, one JSON
answer per ``execute`` request, a clean exit on ``shutdown``); it is NOT a
database.  This proves the harness itself (process launch, handshake, request
feeding, line-by-line reading, comparison, timeout and exit status handling)
without depending on the real MiniVectorDB engine.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TOOLS = _REPO_ROOT / "tools"
_CASES = _REPO_ROOT / "mysqltest" / "cases"

FAKE_PROGRAM_TEMPLATE = '''
import json
import sys

RESPONSES = __PAYLOAD__

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

sys.stdout.write(json.dumps({"type": "ready"}) + "\\n")
sys.stdout.flush()
sys.stderr.write("fake-db: diagnostics belong on stderr\\n")
sys.stderr.flush()

for _line in sys.stdin:
    _line = _line.strip()
    if not _line:
        continue
    try:
        _request = json.loads(_line)
    except ValueError:
        continue
    _id = _request.get("id")
    if _request.get("op") == "shutdown":
        sys.stdout.write(json.dumps({"id": _id, "ok": True}) + "\\n")
        sys.stdout.flush()
        break
    if RESPONSES:
        _payload = RESPONSES.pop(0)
    else:
        _payload = {"error": {"code": "no_canned_response", "message": "no canned response"}}
    _response = {"id": _id, "ok": True}
    _response.update(_payload)
    sys.stdout.write(json.dumps(_response, ensure_ascii=False) + "\\n")
    sys.stdout.flush()
'''


def _write_fake_program(directory: Path, responses) -> Path:
    """Write the fake database program and return its path."""
    script = directory / "fake_db.py"
    payload = repr(responses)
    script.write_text(
        FAKE_PROGRAM_TEMPLATE.replace("__PAYLOAD__", payload), encoding="utf-8"
    )
    return script


class HarnessSmokeTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="minivectordb-harness-")
        self.work = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.expected = json.loads(
            (_CASES / "sql_crud.result").read_text(encoding="utf-8")
        )

    def _run_harness(self, program, extra_args=()):
        command = [
            sys.executable,
            str(_TOOLS / "run_case.py"),
            "--case",
            "sql_crud",
            "--program",
            program,
            "--timeout",
            "20",
        ]
        command.extend(extra_args)
        return subprocess.run(
            command,
            cwd=str(_REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            timeout=180,
        )

    def test_matching_responses_pass(self):
        program = _write_fake_program(self.work, self.expected)
        completed = self._run_harness(
            '"{}" "{}"'.format(sys.executable, program)
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg="stdout:\n{}\nstderr:\n{}".format(
                completed.stdout, completed.stderr
            ),
        )
        self.assertIn("[PASS] sql_crud", completed.stdout)
        self.assertIn("statements=7", completed.stdout)
        self.assertIn("summary: 1/1 cases passed", completed.stdout)

    def test_mismatching_response_is_reported(self):
        broken = json.loads(json.dumps(self.expected))
        # same number of rows, but the row carries another content
        broken[2]["rows"][0] = [1, "修改"]
        program = _write_fake_program(self.work, broken)
        completed = self._run_harness(
            '"{}" "{}"'.format(sys.executable, program)
        )
        self.assertEqual(completed.returncode, 1, msg=completed.stdout)
        self.assertIn("[FAIL] sql_crud", completed.stdout)
        self.assertIn("statement 3", completed.stdout)
        self.assertIn("rows:", completed.stdout)
        self.assertIn("summary: 0/1 cases passed", completed.stdout)

    def test_verbose_prints_statements(self):
        program = _write_fake_program(self.work, self.expected)
        completed = self._run_harness(
            '"{}" "{}"'.format(sys.executable, program), ["--verbose"]
        )
        self.assertEqual(completed.returncode, 0, msg=completed.stdout)
        self.assertIn("--- statement #1: CREATE TABLE docs", completed.stdout)
        self.assertIn("--- answer #1:", completed.stdout)

    def test_program_that_exits_early_fails_the_case(self):
        script = self.work / "broken_db.py"
        script.write_text("raise SystemExit(3)\n", encoding="utf-8")
        completed = self._run_harness(
            '"{}" "{}"'.format(sys.executable, script)
        )
        self.assertEqual(completed.returncode, 1, msg=completed.stdout)
        self.assertIn("[FAIL] sql_crud", completed.stdout)
        self.assertTrue(
            "exited" in completed.stdout or "stopped accepting input" in completed.stdout,
            msg=completed.stdout,
        )

    def test_program_that_times_out_fails_the_case(self):
        script = self.work / "slow_db.py"
        script.write_text(
            "import sys\n"
            "sys.stdout.write('{\"type\":\"ready\"}\\n')\n"
            "sys.stdout.flush()\n"
            "for line in sys.stdin:\n"
            "    pass\n",
            encoding="utf-8",
        )
        completed = self._run_harness(
            '"{}" "{}"'.format(sys.executable, script), ["--timeout", "2"]
        )
        self.assertEqual(completed.returncode, 1, msg=completed.stdout)
        self.assertIn("[FAIL] sql_crud", completed.stdout)
        self.assertIn("timeout", completed.stdout)

    def test_unknown_case_is_rejected(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(_TOOLS / "run_case.py"),
                "--case",
                "does_not_exist",
            ],
            cwd=str(_REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            timeout=60,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("unknown public case", completed.stderr)


if __name__ == "__main__":
    unittest.main()
