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

Rows are individual JSON files. A successful statement first stages and validates every operation, then publishes transaction overrides; statement validation failure leaves earlier successful statements intact. COMMIT writes a redo file, fsyncs it, atomically publishes it and fsyncs the directory before applying materialized row changes. Recovery replays published redo records idempotently, ignoring incomplete temporary files. Uncommitted workspace files are discarded. Catalog and committed rows are reconstructed from redo records.

The vector index is a custom disk-resident sorted first-coordinate projection with binary search and an exact lower bound. Querying expands around the query coordinate, filters candidates before ranking, computes full Euclidean distance, and breaks equal-distance ties by ascending primary key. It stops only after the lower bound excludes further TopK improvements. ANN mode currently uses this exact algorithm, giving exact recall rather than approximate acceleration. Transactional queries use an overlay-aware scan. Committed mutations invalidate indexes; they are rebuilt before use. Index construction and unlimited ORDER BY use external runs with bounded merge fan-in (64).

## Validation and limits

Windows local validation reused the unchanged official public SQL and result assertions: 20 cases and 95 SQL assertions passed. The official POSIX driver could not run on this Windows host; this is NOT a Docker or formal evaluation pass. Independent checks compare 175 filtered TopK results against float32 reference calculations and verify five forced-termination recoveries, transaction rollback and failed-statement atomicity.

```sh
python check_windows.py
python check_recovery.py
python protocol_check/run.py --image minivectordb:local --output protocol-results/public
```

The first two scripts use temporary directories and do not require Docker. The final command requires Linux/Docker and has not yet been verified here. Resource limits, data larger than RAM, formal 29-case admission and competition performance have NOT been measured. Each JSON request and response is materialized in memory. Row scans are streamed, TopK retains K rows, and external sorting limits run size; these design properties are not a substitute for a resource-limit test. Recovery replays all historical commits without checkpoint compaction, and the coordinate index can approach a full scan in high dimensions. Per-row filesystem overhead may limit performance. Do not infer a ranking or full score from the public results.
