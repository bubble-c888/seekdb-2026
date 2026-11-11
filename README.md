# MiniVectorDB current implementation

See [MINIVECTORDB.md](MINIVECTORDB.md) for the current entry point, storage design, build instructions and known limits. Docker runs main.py. The pre-existing engine.py, storage.py, vector_index.py and minivectordb/ remain as historical implementations and are not imported by the current service.

# MiniVectorDB

The database runs as a single Python 3.12 process in a Linux x86_64 container. It uses only the Python standard library.

```sh
docker build --platform linux/amd64 -t minivectordb-dev .
docker volume create minivectordb-demo
docker run --rm -i --mount type=volume,src=minivectordb-demo,dst=/data minivectordb-dev
```

Use a new volume for a clean database. After the process emits its ready object, send one UTF-8 JSON request per line and wait for its response. Send these requests in order:

```jsonl
{"id":1,"op":"execute","sql":"CREATE TABLE docs (id BIGINT PRIMARY KEY, v VECTOR(2));","search_mode":"exact"}
{"id":2,"op":"execute","sql":"BEGIN;","search_mode":"exact"}
{"id":3,"op":"execute","sql":"INSERT INTO docs VALUES (1, '[3,4]');","search_mode":"exact"}
{"id":4,"op":"execute","sql":"COMMIT;","search_mode":"exact"}
{"id":5,"op":"execute","sql":"BEGIN;","search_mode":"exact"}
{"id":6,"op":"execute","sql":"DELETE FROM docs WHERE id = 1;","search_mode":"exact"}
{"id":7,"op":"execute","sql":"ROLLBACK;","search_mode":"exact"}
{"id":8,"op":"shutdown"}
```

Start a new container with the same volume. After ready, `SELECT id FROM docs;` returns row 1. Send this request to check the vector result:

```jsonl
{"id":9,"op":"execute","sql":"SELECT id, L2_DISTANCE(v, '[0,0]') AS distance FROM docs ORDER BY distance LIMIT 1;","search_mode":"exact"}
```

The result is `[[1, 5.0]]`. SQL errors return `ok: false` and a semantic error code. Diagnostics use stderr.

The SQL subset includes one BIGINT primary key per table, VARCHAR and VECTOR columns, CRUD, AND filters, ORDER BY, LIMIT, L2_DISTANCE, vector index creation and single-client transactions. Ordered ties use the primary key. Table definitions and index definitions are stored in a catalog. Each row has a file under its table directory. Changes are staged under `/data/pending`; commit synchronizes the staged data and renames that directory to `committed` before applying changes. Startup replays committed changes and discards uncommitted staging before reporting ready.

Bounded TopK queries stream rows and retain only the best candidates. Small vector indexes use a bounded vantage-point tree rebuilt from committed rows at startup and refreshed after mutations. Tables outside the index cache budget use streaming exact search. Queries with at least 32 dimensions also use exact search because the bounded tree search missed the recall target on a local 128-dimensional sample. UPDATE and DELETE spool matching primary keys to disk before changing rows. Unbounded query responses are collected in memory. The row-per-file I/O cost and the official performance workload remain areas for further testing. Use exact mode for the initial performance submission configuration.

`python check_container.py` runs a local Docker smoke test that includes restart and rollback. On Windows, pass `--package` with the path to the official `mysqltest` directory to check all public cases in Docker. This local runner sends requests in batches. Use the official Linux protocol checker to verify interactive timing.
