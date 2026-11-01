# AMD GPU CAGRA build acceleration: verification and report

This document defines the paired benchmark used for the AMD contest submission.
Official scoring is performed by the contest evaluator with the frozen V5
baseline; local numbers are diagnostic only.

## Candidate profile

| Input profile | Graph build | Search dataset | Graph degree | Intermediate degree | NN-Descent iterations |
| --- | --- | --- | ---: | ---: | ---: |
| `n >= 100000`, `dim <= 128` | FP16 NN-Descent | FP32 CAGRA | 128 | 160 | 32 |
| larger-dimensional inputs | FP16 NN-Descent | FP32 CAGRA | 160 | 160 | 40 |

The final searchable graph always uses hipVS CAGRA search. FP16 is internal to
graph construction only; IDs, persisted vectors, queries, and returned
distances remain FP32.

## Fixed evaluation contract

- datasets: SIFT250K (`250K x 128`, weight 40%) and Hidden250K
  (`250K x 256`, weight 60%);
- metric: L2;
- DDL:
  `CREATE VECTOR INDEX idx_items ON items(v) WITH (distance=l2, type=hnsw, lib=cuvs, m=16, ef_construction=200, ef_search=64);`
- query:
  `SELECT id FROM items ORDER BY l2_distance(v, <query>) APPROXIMATE LIMIT <k>;`
- build timer starts at DDL and ends after the first legal query;
- query benchmark uses 4096 single-row queries for both Top-10 and Top-100.

## Reproducible procedure

1. Check out the frozen baseline and the candidate in separate directories.
2. In each directory, build the bridge and seekdb with the exact V5 commands:

   ```bash
   HIPVS_ROOT=/opt/hipvs ROCM_ROOT=/opt/rocm \
     tools/hipvs_bridge/build.sh /tmp/bridge/libseekdb_cuvs_bridge.so
   cmake -S . -B build_release -G "Unix Makefiles" \
     -DCMAKE_BUILD_TYPE=RelWithDebInfo \
     -DOB_ENABLE_UNITY=ON \
     -DOB_USE_LLD=ON \
     -DOB_BUILD_CUVS=ON \
     -DOB_BUILD_CUVS_TRACE=OFF \
     -DCUVS_BRIDGE_LIB=/tmp/bridge/libseekdb_cuvs_bridge.so
   cmake --build build_release --target seekdb
   ```

3. Start seekdb with an empty base directory and import each dataset into a
   table without a vector index.
4. Alternate baseline/candidate order across the two datasets.
5. For each run, record:
   - DDL + first-query wall time;
   - Recall@10 and Recall@100 against exact L2 ground truth;
   - 4096-query Top-10 and Top-100 QPS;
   - peak GPU/host memory;
   - profiler evidence that CAGRA build and CAGRA search ran on the GPU.
6. Reject a pairing if GPU clocks, system load, profiler samples, or result
   cardinality are abnormal, then rerun that pairing.

## Result table

Fill this table from one uninterrupted official-equivalent run. Do not copy
reference numbers from the contest page.

| Dataset | Baseline build (s) | Candidate build (s) | Build speedup | Candidate Recall@10 | Candidate Recall@100 | Top-10 QPS ratio | Top-100 QPS ratio | Peak VRAM | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| SIFT250K | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Hidden250K | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## Synthetic bridge diagnostic

A same-host bridge diagnostic on the contest Radeon instance used 250,000
deterministic FP32 vectors. This is a pre-screen, not the official score.

| Dim | Baseline build | Candidate build | Build speedup | Baseline Recall@100 | Candidate Recall@100 | Candidate QPS vs baseline |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 41.89–42.83 s | 14.2–15.6 s | approximately 2.7x | 0.9856–0.9875 | 0.9850–0.9925 | approximately 1.4x |
| 256 | 42.84–43.56 s | 16.2–18.5 s | approximately 2.5x | 0.9569–0.9600 | 0.9470–0.9628 | approximately 1.2x |

The 256-dimensional path uses a wider final graph rather than the frozen V5
parameters. Its synthetic result is close to the paired baseline and remains
subject to official Recall validation. All timings show notable GPU warm-up
variance, so the official paired evaluator remains authoritative.

## Acceptance gates

- SIFT250K Recall@10 and Recall@100 are each at least 0.95 and no more than
  0.005 below the paired baseline.
- Hidden250K Recall@10 is at least 0.90, Recall@100 is at least 0.85, and each
  is no more than 0.005 below the paired baseline.
- Top-10/Top-100 query throughput each remains at least 90% of baseline.
- Every query returns `k` unique valid IDs.
- GPU CAGRA build/search is observed; no CPU fallback or prebuilt index is
  used.
- Peak VRAM is within the physical quota and 1.25x of the paired baseline.

## Local notes

The implementation in this branch cannot be GPU-validated on a machine without
ROCm and a supported Radeon device. Run the procedure above in the AMD Radeon
Cloud `seekdb-hipvs` environment before submitting a scored attempt.
