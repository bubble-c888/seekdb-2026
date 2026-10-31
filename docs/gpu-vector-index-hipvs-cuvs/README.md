# hipVS/cuVS GPU vector backend

Status: **Phase 1 implemented and review-gated** on AMD `gfx1100` with ROCm
7.2.4. This directory documents the current integration and its reproducible
validation assets. Historical feasibility experiments remain available in Git
history rather than in the product-facing tree.

## Scope

The backend adds optional hipVS/cuVS acceleration behind seekdb's existing
vector adaptor. It does not replace VSAG globally.

- GPU support is compiled only with `OB_BUILD_CUVS=ON`; the default is `OFF`.
- A dense L2 HNSW/HGRAPH index opts in with `WITH (lib=cuvs)`.
- `lib=vsag` indexes keep their existing CPU behavior.
- Cosine and inner-product metrics are rejected for `lib=cuvs` at DDL time.
- Filters, deletes, stale GPU indexes, and unavailable GPUs fall back to VSAG.
- `dbms_vector.batch_knn` exposes one-call batched GPU ANN for explicit batch
  workloads.

The product integration lives in:

- [`ob_vsag_adaptor.cpp`](../../src/oblib/lib/vector/ob_vsag_adaptor.cpp):
  per-index marking, buffered vectors, lazy CAGRA build/search, freshness and
  safety fallback, trace points, and batch entry points.
- [`seekdb_cuvs_bridge.c`](../../tools/hipvs_bridge/seekdb_cuvs_bridge.c):
  persistent query workspaces, CAGRA parameter selection, graph-only snapshot
  serialization, and C-ABI lifecycle management.
- [`seekdb_cuvs_fp16_build.cpp`](../../tools/hipvs_bridge/seekdb_cuvs_fp16_build.cpp):
  optional FP16 graph construction followed by reconstruction of the
  searchable index over the original FP32 vectors.
- [`ob_plugin_vector_index_adaptor.cpp`](../../src/observer/vector_index/ob_plugin_vector_index_adaptor.cpp):
  maps declarative `lib=cuvs` indexes to the adaptor handle.
- [`ob_vector_index_util.cpp`](../../src/observer/vector_index/ob_vector_index_util.cpp):
  validates the L2-only DDL contract.
- [`ob_dbms_vector_mysql.cpp`](../../src/pl/sys_package/ob_dbms_vector_mysql.cpp):
  implements `dbms_vector.batch_knn`.

## Bridge

seekdb links a small C ABI bridge instead of importing hipVS/cuVS C++ types into
the server. The maintained bridge source and build entry point are under
[`tools/hipvs_bridge`](../../tools/hipvs_bridge/):

```bash
tools/hipvs_bridge/build.sh /work/bridge/libseekdb_cuvs_bridge.so
```

The bridge exports the build/search/free/persistence C ABI. Query workspaces
and CAGRA search parameters are created once per index and reused across calls;
single-row SQL searches therefore avoid three RMM allocations and repeated
parameter allocation on every query.

Snapshot serialization stores the CAGRA graph separately from the outer
snapshot's FP32 vectors and external IDs. Restore combines the two instead of
writing the vector dataset twice.

## AMD build acceleration

For large, lower-dimensional inputs (`n >= 100000`, `dim <= 128`), the bridge
builds the NN-Descent graph through an FP16 host view, copies the resulting
graph back, and reconstructs the searchable CAGRA index with the original FP32
dataset. This is the SIFT-like path:

- graph degree: 128;
- intermediate graph degree: 160;
- NN-Descent iterations: 32;
- graph construction precision: FP16;
- search width: 8;
- search dataset and distances: FP32.

Higher-dimensional inputs use a wider degree-160 graph with a degree-192
intermediate graph, 48 NN-Descent iterations, FP16 graph construction, and
width-8 FP32 search. This recovers Recall while keeping build and query work
below the frozen V5 baseline.
The heuristic depends only on input size and dimension; it does not inspect
dataset names or evaluation metadata.

Build seekdb with the resulting library:

```bash
./build.sh release \
  -DOB_BUILD_CUVS=ON \
  -DOB_BUILD_CUVS_TRACE=OFF \
  -DCUVS_BRIDGE_LIB=/work/bridge/libseekdb_cuvs_bridge.so \
  --make -j64
```

For a CPU-only build, leave the defaults or pass:

```bash
./build.sh release \
  -DOB_BUILD_CUVS=OFF \
  -DOB_BUILD_CUVS_TRACE=OFF \
  --make -j64
```

## SQL usage

Per-index GPU opt-in:

```sql
CREATE TABLE items(
  id BIGINT PRIMARY KEY,
  embedding VECTOR(128),
  VECTOR INDEX idx_embedding(embedding)
    WITH (distance=l2, type=hnsw, lib=cuvs)
);
```

Explicit batch ANN writes results to a caller-created output table:

```sql
CALL dbms_vector.batch_knn(
  "items",
  "query_vectors",
  10,
  "batch_results"
);
```

See [`examples/batch_knn.sql`](examples/batch_knn.sql) for the table contract.

## Validation

The paired contest build benchmark and the report fields are documented in
[`BUILD_BENCHMARK.md`](BUILD_BENCHMARK.md). On a Radeon host, the bridge build,
Top-10/Top-100 buffer reuse, and graph-only snapshot round trip can be checked
with:

```bash
tools/hipvs_bridge/run_persistence_test.sh /tmp/seekdb-cuvs-persistence-test
```

The self-contained GPU smoke runner generates deterministic vectors and ground
truth; it does not use private `/work/datasets` or `/work/bench` assets:

```bash
LD_LIBRARY_PATH=/work/bridge:/opt/hipvs/lib:/opt/rocm/lib \
  docs/gpu-vector-index-hipvs-cuvs/validation/run_gpu_smoke.sh \
    --observer /path/to/trace-build/seekdb \
    --bridge /work/bridge/libseekdb_cuvs_bridge.so \
    --base-dir /tmp/seekdb-cuvs-smoke \
    --port 2981 \
    --render /dev/dri/renderD135 \
    --evidence /tmp/seekdb-cuvs-evidence
```

The runner validates:

- cuVS L2 build/serve and same-process VSAG isolation;
- non-L2 DDL rejection;
- filter, delete, and freshness fallback;
- threshold-triggered CAGRA rebuild;
- one-call `batch_knn`, row cardinality, and recall for Top-10 and Top-100;
- GPU device file descriptors and hidden-GPU fallback;
- bounded observer cleanup and closed ports.

The runner writes a machine-readable `summary.json` into its evidence directory.
Results for a specific candidate belong in that pull request's description or a
validation comment rather than in the source tree.
The no-GPU DDL contract remains covered by
[`vector_index_cuvs_ddl.test`](../../tools/deploy/mysql_test/test_suite/vector_index/t/vector_index_cuvs_ddl.test).

## Current limitations

- Phase 1 supports dense float32 L2 indexes only.
- GPU indexes are process-local and rebuilt lazily.
- Graph-only snapshots require the FP32 vector payload emitted by seekdb's
  outer snapshot.
- The explicit batch API rebuilds its CAGRA index per call.
- GPU validation is manual because upstream CI has no compatible GPU runner.
- Multi-GPU, long-duration, and snapshot-reload coverage are out of scope for
  this phase.
