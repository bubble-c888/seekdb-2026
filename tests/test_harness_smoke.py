#!/usr/bin/env python3
"""End-to-end smoke test of the mysqltest harness.

The test drives ``tools/run_case.py`` against a tiny fake database program that
is written to a temporary file on the fly.  The fake only implements the
response protocol (one JSON line per statement, clean exit on EOF); it is NOT a
database.  This proves the harness itself (process launch, statement feeding,
line-by-line reading, comparison, exit status handling) without depending on
the real MiniVectorDB engine.
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

FAKE_PROGRAM_TEMPLATE = '''\
import json
import sys

RESPONSES = __PAYLOAD__

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

sys.stderr.write("fake-db: diagnostics belong on stderr\\n")
sys.stderr.flush()

for _line in sys.stdin:
    if RESPONSES:
        response = RESPONSES.pop(0)
    else:
        response = {"error": "no canned response", "code": 1064, "sqlstate": "42000"}
    sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\\n")
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
            "--data-dir",
            str(self.work / "cases"),
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
        self.assertIn("statements=9/9", completed.stdout)
        self.assertIn("summary: 1/1 cases passed", completed.stdout)

    def test_mismatching_response_is_reported(self):
        broken = json.loads(json.dumps(self.expected))
        # same number of rows, but the last row carries another category
        broken[2]["rows"][2] = [3, "甲", "三"]
        program = _write_fake_program(self.work, broken)
        completed = self._run_harness(
            '"{}" "{}"'.format(sys.executable, program)
        )
        self.assertEqual(completed.returncode, 1, msg=completed.stdout)
        self.assertIn("[FAIL] sql_crud", completed.stdout)
        self.assertIn("statement #3", completed.stdout)
        self.assertIn("row 2 mismatch", completed.stdout)
        self.assertIn("column 1 mismatch", completed.stdout)
        self.assertIn("expected:", completed.stdout)
        self.assertIn("actual:", completed.stdout)
        self.assertIn("summary: 0/1 cases passed", completed.stdout)

    def test_verbose_prints_statements(self):
        program = _write_fake_program(self.work, self.expected)
        completed = self._run_harness(
            '"{}" "{}"'.format(sys.executable, program), ["--verbose"]
        )
        self.assertEqual(completed.returncode, 0, msg=completed.stdout)
        self.assertIn("--- statement #1: CREATE TABLE docs", completed.stdout)
        self.assertIn("expected:", completed.stdout)

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
        self.assertIn("unknown case", completed.stderr)


if __name__ == "__main__":
    unittest.main()
