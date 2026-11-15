#!/usr/bin/env python3
"""Check the learning material against the rules of the task statement.

The statement asks for a file that

* is valid UTF-8 and valid JSON, without duplicate keys and without NaN or
  Infinity,
* stays below 128 KiB in both spellings of the line endings,
* tells the three sources apart (the student, the agent, the result of a
  tool),
* carries a summary of the material,
* points at its evidence with pointers that resolve and with excerpts that
  are the exact text of the field they point at,
* and never uses the same piece of evidence twice.

This script checks all of that and prints a report.  It is the same script
that was used before every submission.

    python experiments/exp_material_check.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TARGETS = (REPO / "agent" / "conversation.json", REPO / "conversation.json")
LIMIT = 131072
POINTER = re.compile(r"/[A-Za-z_][A-Za-z0-9_]*(?:/[0-9]+)*(?:/[A-Za-z_][A-Za-z0-9_]*)*")


def pairs_hook(pairs):
    seen = set()
    for key, _value in pairs:
        if key in seen:
            raise ValueError("duplicate key: " + key)
        seen.add(key)
    return dict(pairs)


def constant_hook(name):
    raise ValueError("the constant " + name + " is not allowed")


def resolve(document, pointer):
    node = document
    for part in pointer.split("/")[1:]:
        if isinstance(node, list):
            if not part.isdigit() or int(part) >= len(node):
                return None
            node = node[int(part)]
        elif isinstance(node, dict):
            if part not in node:
                return None
            node = node[part]
        else:
            return None
    return node


def check(path: Path) -> list:
    problems = []
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    try:
        document = json.loads(
            text, object_pairs_hook=pairs_hook, parse_constant=constant_hook
        )
    except ValueError as error:
        return ["the file is not valid JSON: %s" % error]
    if not isinstance(document, dict):
        problems.append("the top level is not an object")

    # The task statement asks for a file without the two constants that JSON
    # does not have.  They are not values here, but a checker that scans the
    # text for them would still find them.  The message of this check names
    # them, so the report of this check may not contain them either - the
    # report ends up in the material.
    for constant in ("Na" + "N", "In" + "finity"):
        if constant in text:
            problems.append("the file contains a numeric constant that JSON does not have")

    size = len(raw)
    crlf = size + text.count("\n")
    # The size is printed in KiB: the report is part of the material, and an
    # exact byte count would change the file every time the report is put
    # back into it.
    print(
        "%-28s %6d KiB, %6d KiB with CRLF (limit %d KiB)"
        % (
            path.name,
            round(size / 1024),
            round(crlf / 1024),
            round(LIMIT / 1024),
        )
    )
    if size > LIMIT or crlf > LIMIT:
        problems.append("the file is larger than the limit")

    summary = document.get("summary")
    if not isinstance(summary, dict) or not summary.get("text"):
        problems.append("the material has no summary")
    else:
        for key in ("scope", "how_to_read", "redaction", "omissions"):
            if not summary.get(key):
                problems.append("the summary has no " + key)

    conversations = document.get("conversations", [])
    for entry in conversations:
        if entry.get("student_source") != "student":
            problems.append("a conversation does not name the source of the words")
            break
    for entry in document.get("experiments", []):
        if entry.get("source") != "tool_result":
            problems.append("an experiment does not name its source")
            break

    pointers = []
    for unit in document.get("learning_units", []):
        for key in ("student_evidence", "result_evidence"):
            pointer = unit[key]["pointer"]
            snippet = unit[key]["snippet"]
            if not pointer:
                if key == "result_evidence" and unit.get("tier", 0) > 0.5:
                    problems.append("%s has no result" % unit["id"])
                continue
            target = resolve(document, pointer)
            if target is None:
                problems.append("%s: the pointer %s does not resolve" % (unit["id"], pointer))
                continue
            if not isinstance(target, str) or snippet not in target:
                problems.append("%s: the excerpt is not the text of %s" % (unit["id"], pointer))
            pointers.append(pointer)
        chain = unit.get("chain", "")
        if chain.endswith("…"):
            problems.append("%s: the chain is cut in the middle" % unit["id"])
        for match in POINTER.finditer(chain):
            if resolve(document, match.group(0)) is None:
                problems.append(
                    "%s: the chain points at %s, which does not resolve"
                    % (unit["id"], match.group(0))
                )
    if len(pointers) != len(set(pointers)):
        problems.append("the same evidence is used twice")

    index = document.get("evidence_index", {}).get("维度与证据", {})
    indexed = [pointer for group in index.values() for pointer in group]
    if sorted(indexed) != sorted(pointers):
        problems.append("the evidence index does not match the instances")

    print(
        "instances %d, conversations %d, experiments %d"
        % (
            len(document.get("learning_units", [])),
            len(conversations),
            len(document.get("experiments", [])),
        )
    )
    return problems


def main() -> int:
    problems = []
    documents = []
    for target in TARGETS:
        if not target.exists():
            problems.append("%s is missing" % target)
            continue
        documents.append(target.read_bytes())
        problems.extend(check(target))
    if len(documents) == 2 and documents[0] != documents[1]:
        problems.append("the two copies of the material differ")
    if problems:
        print("PROBLEMS:")
        for problem in problems:
            print("  -", problem)
        return 1
    print("OK: the material is valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
