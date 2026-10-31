#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUTPUT_DIR=${1:-/tmp/seekdb-cuvs-persistence-test}
HIPVS_ROOT=${HIPVS_ROOT:-/opt/hipvs}
ROCM_ROOT=${ROCM_ROOT:-/opt/rocm}
HIPCC=${HIPCC:-$ROCM_ROOT/bin/hipcc}

mkdir -p "$OUTPUT_DIR"
HIPVS_ROOT="$HIPVS_ROOT" ROCM_ROOT="$ROCM_ROOT" HIPCC="$HIPCC" \
  "$SCRIPT_DIR/build.sh" "$OUTPUT_DIR/libseekdb_cuvs_bridge.so"

"$HIPCC" \
  -std=c++17 \
  -O2 \
  -I"$SCRIPT_DIR" \
  "$SCRIPT_DIR/test_persistence.cpp" \
  -L"$OUTPUT_DIR" \
  -Wl,-rpath,"$OUTPUT_DIR:$HIPVS_ROOT/lib:$ROCM_ROOT/lib" \
  -lseekdb_cuvs_bridge \
  -o "$OUTPUT_DIR/test_persistence"

export LD_LIBRARY_PATH="$OUTPUT_DIR:$HIPVS_ROOT/lib:$ROCM_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$OUTPUT_DIR/test_persistence" write "$OUTPUT_DIR/snapshot.bin"
"$OUTPUT_DIR/test_persistence" read "$OUTPUT_DIR/snapshot.bin"
echo "PASS persistence smoke: $OUTPUT_DIR"
