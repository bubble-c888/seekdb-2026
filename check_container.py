"""Run a database container twice and verify the same volume is recovered."""

import argparse
import json
import math
import subprocess
import uuid
from contextlib import contextmanager
from pathlib import Path


def run(volume, statements, mode="exact"):
    requests = [
        {"id": ident, "op": "execute", "sql": sql, "search_mode": search_mode}
        for ident, (sql, search_mode) in enumerate(
            (item if isinstance(item, tuple) else (item, mode) for item in statements), start=1,
        )
    ]
    requests.append({"id": len(requests) + 1, "op": "shutdown"})
    result = subprocess.run(
        ["docker", "run", "--rm", "-i", "--network=none", "--read-only",
         "--memory=2g", "--cpus=2", "--mount", f"type=volume,src={volume},dst=/data",
         "minivectordb-dev"],
        input="".join(json.dumps(request) + "\n" for request in requests),
        capture_output=True, encoding="utf-8", timeout=90,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"Docker exited with {result.returncode}")
    lines = [json.loads(line) for line in result.stdout.splitlines()]
    if len(lines) != len(requests) + 1 or lines[0] != {"type": "ready"}:
        raise RuntimeError(f"Unexpected protocol output: {result.stdout}")
    for request, response in zip(requests, lines[1:]):
        if response.get("id") != request["id"] or type(response.get("ok")) is not bool:
            raise RuntimeError(f"Unexpected response to {request}: {response}")
    if lines[-1]["ok"] is not True:
        raise RuntimeError(f"Shutdown failed: {lines[-1]}")
    return lines[1:-1]


@contextmanager
def test_volume():
    volume = "minivectordb-check-" + uuid.uuid4().hex
    subprocess.run(["docker", "volume", "create", volume], check=True, capture_output=True)
    try:
        yield volume
    finally:
        cleanup = subprocess.run(["docker", "volume", "rm", volume], capture_output=True, text=True)
        if cleanup.returncode:
            print(f"Could not remove test volume {volume}: {cleanup.stderr.strip()}")


def smoke():
    with test_volume() as volume:
        results = run(volume, [
            "CREATE TABLE docs (id BIGINT PRIMARY KEY, embedding VECTOR(2));",
            "BEGIN;",
            "INSERT INTO docs VALUES (1, '[3,4]');",
            "COMMIT;",
            "BEGIN;",
            "INSERT INTO docs VALUES (2, '[0,0]');",
        ])
        if (not all(response["ok"] for response in results)
                or results[2].get("affected_rows") != 1 or results[5].get("affected_rows") != 1):
            raise RuntimeError(f"Unexpected insert results: {results}")
        result, = run(volume, [
            "SELECT id, L2_DISTANCE(embedding, '[0,0]') AS distance FROM docs ORDER BY distance;",
        ])
        if result.get("columns") != ["id", "distance"] or result.get("rows") != [[1, 5]]:
            raise RuntimeError(f"Unexpected result after restart: {result}")
        print("PASS: Docker startup, vector query, committed data recovery, and shutdown rollback")


def equal(actual, expected):
    if isinstance(expected, list):
        return (isinstance(actual, list) and len(actual) == len(expected)
                and all(equal(a, b) for a, b in zip(actual, expected)))
    if isinstance(expected, float):
        return isinstance(actual, (int, float)) and math.isclose(actual, expected, abs_tol=1e-5, rel_tol=1e-5)
    return type(actual) is type(expected) and actual == expected


def public_suite(package):
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    total = 0
    for case in manifest["cases"]:
        expected = json.loads((package / case["result"]).read_text(encoding="utf-8"))
        statements = (package / case["test"]).read_text(encoding="utf-8").splitlines()
        responses, batch = [], []
        mode = "exact"
        with test_volume() as volume:
            for statement in filter(str.strip, statements):
                if statement == "--restart":
                    responses.extend(run(volume, batch))
                    batch.clear()
                elif statement.startswith("--mode "):
                    mode = statement.split()[1]
                else:
                    batch.append((statement, mode))
            responses.extend(run(volume, batch))
        if len(responses) != len(expected):
            raise RuntimeError(f"{case['id']}: expected {len(expected)} SQL responses, got {len(responses)}")
        for number, (response, want) in enumerate(zip(responses, expected), start=1):
            if "error" in want:
                passed = not response["ok"] and response.get("error", {}).get("code") == want["error"]
            else:
                passed = response["ok"]
                for field in want.keys() - {"unordered"}:
                    actual, value = response.get(field), want[field]
                    if field == "rows" and want.get("unordered") and isinstance(actual, list):
                        actual, value = sorted(actual), sorted(value)
                    passed = passed and equal(actual, value)
            if not passed:
                raise RuntimeError(f"{case['id']} SQL {number}: expected {want}, got {response}")
        total += len(expected)
        print(f"PASS: {case['id']} ({len(expected)} SQL)", flush=True)
    print(f"PASS: {len(manifest['cases'])} public container cases, {total} SQL")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, help="Directory containing the official mysqltest manifest")
    args = parser.parse_args()
    if args.package:
        public_suite(args.package)
    else:
        smoke()


if __name__ == "__main__":
    main()
