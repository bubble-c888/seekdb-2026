# MiniVectorDB

Custom single-process database, implemented in Python 3.13 with the standard library only. No database or vector-index libraries are used. Runtime requires no network.

## Build and run

```sh
docker build -t minivectordb:local .
docker run --rm -i --cpus=2 --memory=2g -v minivectordb-data:/data minivectordb:local
```

The service prints one JSONL ready message before reading input. All persistent state, transaction staging files and indexes are under /data. Logs go to stderr. stdin EOF or shutdown rolls back an open transaction.

Example requests (send one JSON object per line):

```json
{"id":1,"op":"execute","sql":"CREATE TABLE docs (id BIGINT PRIMARY KEY, name VARCHAR(40), v VECTOR(2))"}
{"id":2,"op":"execute","sql":"INSERT INTO docs VALUES (1,'first','[3,4]')"}
{"id":3,"op":"execute","sql":"SELECT id FROM docs ORDER BY L2_DISTANCE(v,'[0,0]') LIMIT 1","search_mode":"exact"}
{"id":4,"op":"execute","sql":"BEGIN"}
{"id":5,"op":"execute","sql":"UPDATE docs SET name = 'changed' WHERE id = 1"}
{"id":6,"op":"execute","sql":"ROLLBACK"}
{"id":7,"op":"execute","sql":"BEGIN"}
{"id":8,"op":"execute","sql":"INSERT INTO docs VALUES (2,'second','[0,0]')"}
{"id":9,"op":"execute","sql":"COMMIT"}
{"id":10,"op":"shutdown"}
```

Run the same container again with the same volume to recover committed records. SELECT, WHERE with comparison/AND, ORDER BY, LIMIT, INSERT, UPDATE, DELETE, CREATE TABLE, CREATE VECTOR INDEX and explicit transactions are supported. BIGINT uses signed 64-bit bounds; VARCHAR counts Unicode code points; vector coordinates are validated and rounded to finite float32. Primary keys cannot be updated. Unsupported SQL returns a nonempty error code; the connection remains usable.

## Storage and correctness

Rows are individual JSON files with sparse per-column patch files for updates. Reads merge the original row and its current patches; inserts replace the row and clear prior patches, while deletes remove both. A successful statement first stages and validates every operation, then publishes transaction overrides; statement validation failure leaves earlier successful statements intact. COMMIT writes a redo file, fsyncs it, atomically publishes it and fsyncs the directory before applying materialized row changes. Recovery restores the atomic catalog/applied-sequence checkpoint, then replays later published redo records idempotently, ignoring incomplete temporary files. Checkpoints are published only after complete materialization; interrupted materialization and interrupted recovery are retried from the previous checkpoint. WAL history is retained. Uncommitted workspace files are discarded. Catalog and committed rows are reconstructed from redo records.

The vector index is a custom disk-resident sorted first-coordinate projection with binary search and an exact lower bound. Querying expands around the query coordinate, filters candidates before ranking, computes full Euclidean distance, and breaks equal-distance ties by ascending primary key. It stops only after the lower bound excludes further TopK improvements. ANN mode currently uses this exact algorithm, giving exact recall rather than approximate acceleration. Transactional queries use an overlay-aware scan. Committed mutations invalidate indexes; they are rebuilt before use. Index construction and unlimited ORDER BY use external runs with bounded merge fan-in (64).

## Validation and limits

Windows local validation reused the unchanged official public SQL and result assertions: 20 cases and 95 SQL assertions passed. The official POSIX driver could not run directly on this Windows host. Subsequent GitHub Actions Linux runs built the Docker image and passed the official handshake and public SQL tests. This is not formal 29-case admission. Independent checks compare 175 filtered TopK results against float32 reference calculations and verify five forced-termination recoveries, transaction rollback and failed-statement atomicity.

```sh
python check_windows.py
python check_recovery.py
python protocol_check/run.py --image minivectordb:local --output protocol-results/public
```

The first two scripts use temporary directories and do not require Docker. The final command requires Linux/Docker. It passed on GitHub Actions Linux after the initial Windows checks (runs 42988403016 and 42988666245). A larger-than-memory scenario has subsequently passed under the official Docker transport (2 CPUs, 2 GiB, 60-second requests), as recorded below. Formal 29-case admission and competition performance remain unverified. Each JSON request and response is materialized in memory. Row scans are streamed, TopK retains K rows, and external sorting limits run size; these design properties are not a substitute for a resource-limit test. Historical WAL files are retained without compaction, and the coordinate index can approach a full scan in high dimensions. Per-row filesystem overhead may limit performance. Do not infer a ranking or full score from the public results.

## Timeout fix validation (2026-11-11)

Platform reported one JSON response timeout without identifying the SQL. A parser regression was independently reproduced: copying/scanning the remaining SQL per token caused a 15,480,910-byte statement to require 96.393 seconds to parse; after removing that quadratic work, the same local benchmark required 0.442 seconds. This compares parsing only.

GitHub Actions run 42992902164 built the fixed Docker image, passed official public tests, and executed a 12,000-row 128-dimensional insert through the unchanged official Docker transport under its 60-second request limit and 2-GiB resource setting. The 15,481,803-byte INSERT responded in 8.686 seconds. Filtered indexed TopK and two force-kill recoveries passed; independent smaller recovery and exact TopK checks also passed. These are new local/CI checks, not proof of hidden formal acceptance. Run `python3 check_bulk.py --image minivectordb:local` to reproduce the added regression.

## Larger-than-memory timeout reproduction and fix

The prior parser fix did not resolve the platform's single reported timeout. Without access to hidden inputs, we reproduced the identical JSON response timeout in an independent declared-requirement scenario: 17,000 rows with 131,072-character text fields, totaling 2,228,224,000 logical payload bytes, inside the unchanged 2,147,483,648-byte Docker memory limit. The original full-table scalar update exceeded the 60-second response deadline (run 42997470561).

Updates now stage and redo only changed fields, preserving untouched large text and vector values. Autocommit avoids an unnecessary transaction-overlay copy. An atomic applied checkpoint prevents rewriting already-materialized history on process restart. Neither request deadlines nor resource limits were relaxed.

The same scale scenario passed on Linux Docker (run 42999026249): full-table update 10.599 s; explicit transaction update 8.483 s; COMMIT 4.277 s; successful force-kill/restart; post-recovery query 4.890 s; DELETE 8.585 s. Full public SQL, the 12,000-row 128D bulk regression, independent TopK/recovery checks and interrupted-patch redo regression also passed (run 42999026223). The scale script additionally covers a transaction update/commit, beyond the original failing reproduction.

```sh
python3 check_scale.py --image minivectordb:local
python3 check_patch_recovery.py
```

The interrupted-redo regression kills a process after partial committed patch materialization, interrupts recovery twice, then verifies all original large text and all final values. Existing transaction rollback, delete/reinsert and vector-index maintenance checks remain passing. These tests reproduce and fix a real timeout but do not identify the undisclosed official failing input or guarantee formal acceptance.
