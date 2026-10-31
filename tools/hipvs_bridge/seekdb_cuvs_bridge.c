/* seekdb <-> hipVS bridge: plain-C CAGRA kNN; all cuVS/ROCm deps live here. */
#include "seekdb_cuvs_bridge.h"
#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/cagra.h>
#include <dlpack/dlpack.h>
#ifdef __HIP_PLATFORM_AMD__
#include <cuvs/cuda_runtime.h>
#else
#include <cuda_runtime.h>
#endif
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Built by seekdb_cuvs_fp16_build.cpp. The returned index is a float32 CAGRA
 * index whose graph was constructed from an FP16 view of the same vectors. */
extern int seekdb_cuvs_build_fp16_float_graph(const float* base, long n, long dim,
                                               size_t graph_degree,
                                               size_t intermediate_graph_degree,
                                               size_t nn_descent_niter,
                                               cuvsResources_t res,
                                               cuvsCagraIndex_t* out_index);

typedef struct {
  cuvsResources_t res;
  cuvsCagraIndexParams_t ip;
  cuvsCagraIndex_t idx;
  cuvsCagraSearchParams_t sp;
  long n, dim;
  /* Reused by the common single-query SQL path. Protected by mu because the
   * handle is intentionally opaque to seekdb and may be used by workers. */
  void* query_dev;
  unsigned int* ids_dev;
  float* dist_dev;
  size_t query_capacity;
  size_t ids_capacity;
  size_t dist_capacity;
  pthread_mutex_t mu;
} sk_cuvs_index;

typedef struct {
  size_t graph_degree;
  size_t intermediate_graph_degree;
  size_t nn_descent_niter;
  int fp16_graph;
} sk_build_profile;

static sk_build_profile sk_choose_build_profile(long n, long dim)
{
  sk_build_profile p;
  if (n >= 100000 && dim <= 128) {
    /* Keep the validated final graph degree. FP16 graph construction plus the
     * minimum recommended 1.5x intermediate degree cuts build work while the
     * persisted/searchable index keeps the original FP32 vectors. */
    p.graph_degree = 128;
    p.intermediate_graph_degree = 160;
    p.nn_descent_niter = 32;
    p.fp16_graph = 1;
  } else {
    /* Spend the additional graph degree only on the 256-dimensional workload,
     * where it recovers Recall while retaining a shorter NN-Descent run. */
    p.graph_degree = 160;
    p.intermediate_graph_degree = 192;
    p.nn_descent_niter = 48;
    p.fp16_graph = 1;
  }
  return p;
}

static void sk_init_dl_tensor(DLManagedTensor* tensor, void* data, long n, long dim,
                              DLDeviceType device_type, int bits, DLDataTypeCode code,
                              int64_t* shape, int64_t* strides)
{
  memset(tensor, 0, sizeof(*tensor));
  shape[0] = n;
  shape[1] = dim;
  tensor->dl_tensor.data = data;
  tensor->dl_tensor.device.device_type = device_type;
  tensor->dl_tensor.device.device_id = 0;
  tensor->dl_tensor.ndim = 2;
  tensor->dl_tensor.dtype.code = code;
  tensor->dl_tensor.dtype.bits = (uint8_t)bits;
  tensor->dl_tensor.dtype.lanes = 1;
  tensor->dl_tensor.shape = shape;
  tensor->dl_tensor.strides = strides;
  tensor->dl_tensor.byte_offset = 0;
}

static int sk_create_search_params(sk_cuvs_index* h)
{
  if (h->sp != NULL) { return 0; }
  return cuvsCagraSearchParamsCreate(&h->sp) == CUVS_SUCCESS ? 0 : 1;
}

static int sk_mul_size(size_t a, size_t b, size_t* result)
{
  if (b != 0 && a > ((size_t)-1) / b) { return 1; }
  *result = a * b;
  return 0;
}

static int sk_ensure_device_buffer(void** ptr, size_t* capacity, size_t needed,
                                   size_t element_size, cuvsResources_t res)
{
  int rc = 0;
  if (needed <= *capacity) { return 0; }
  if (needed > ((size_t)-1) / element_size) { return 1; }
  void* replacement = NULL;
  if (cuvsRMMAlloc(res, &replacement, needed * element_size) != CUVS_SUCCESS) {
    rc = 1;
  } else {
    if (*ptr != NULL && *capacity > 0) {
      cuvsRMMFree(res, *ptr, *capacity * element_size);
    }
    *ptr = replacement;
    *capacity = needed;
  }
  return rc;
}

void* seekdb_cuvs_build(const float* base, long n, long dim)
{
  sk_cuvs_index* h = (sk_cuvs_index*)calloc(1, sizeof(*h));
  if (!h) { return NULL; }
  if (pthread_mutex_init(&h->mu, NULL) != 0) { free(h); return NULL; }
  if (cuvsResourcesCreate(&h->res) != CUVS_SUCCESS) { goto fail_mu; }

  const sk_build_profile profile = sk_choose_build_profile(n, dim);
  if (cuvsCagraIndexParamsCreate(&h->ip) != CUVS_SUCCESS) { goto fail_res; }
  h->ip->build_algo = NN_DESCENT;
  h->ip->graph_degree = profile.graph_degree;
  h->ip->intermediate_graph_degree = profile.intermediate_graph_degree;
  h->ip->nn_descent_niter = profile.nn_descent_niter;

  int64_t ds[2] = {n, dim};
  int64_t strides[2] = {0, 0};
  DLManagedTensor dset;
  sk_init_dl_tensor(&dset, (void*)base, n, dim, kDLCPU, 32, kDLFloat, ds, strides);

  if (profile.fp16_graph) {
    if (seekdb_cuvs_build_fp16_float_graph(base, n, dim, profile.graph_degree,
                                           profile.intermediate_graph_degree,
                                           profile.nn_descent_niter, h->res, &h->idx) != 0) {
      goto fail_ip;
    }
  } else {
    if (cuvsCagraIndexCreate(&h->idx) != CUVS_SUCCESS) { goto fail_ip; }
    if (cuvsCagraBuild(h->res, h->ip, &dset, h->idx) != CUVS_SUCCESS) { goto fail_idx; }
  }
  if (sk_create_search_params(h) != 0) { goto fail_idx; }
  h->n = n;
  h->dim = dim;
  return h;

fail_idx:
  if (h->idx != NULL) { cuvsCagraIndexDestroy(h->idx); }
fail_ip:
  cuvsCagraIndexParamsDestroy(h->ip);
fail_res:
  cuvsResourcesDestroy(h->res);
fail_mu:
  pthread_mutex_destroy(&h->mu);
  free(h);
  fprintf(stderr, "[seekdb_cuvs] build failed: %s\n", cuvsGetLastErrorText());
  return NULL;
}

int seekdb_cuvs_search(void* handle, const float* query, long nq, long topk,
                       unsigned int* out_ids, float* out_dist)
{
  sk_cuvs_index* h = (sk_cuvs_index*)handle;
  if (!h || !query || !out_ids || nq <= 0 || topk <= 0 || h->n <= 0 || h->dim <= 0) {
    return 1;
  }

  int rc = 0;
  pthread_mutex_lock(&h->mu);
  if (sk_create_search_params(h) != 0) { rc = 14; goto done; }
  size_t query_needed = 0;
  size_t output_needed = 0;
  size_t required_itopk = 0;
  if (sk_mul_size((size_t)nq, (size_t)h->dim, &query_needed) != 0 ||
      sk_mul_size((size_t)nq, (size_t)topk, &output_needed) != 0 ||
      sk_mul_size((size_t)topk, 4, &required_itopk) != 0) {
    pthread_mutex_unlock(&h->mu);
    return 18;
  }
  if (sk_ensure_device_buffer(&h->query_dev, &h->query_capacity, query_needed,
                              sizeof(float), h->res) != 0 ||
      sk_ensure_device_buffer((void**)&h->ids_dev, &h->ids_capacity, output_needed,
                              sizeof(unsigned int), h->res) != 0 ||
      sk_ensure_device_buffer((void**)&h->dist_dev, &h->dist_capacity, output_needed,
                              sizeof(float), h->res) != 0) {
    rc = 10;
    goto done;
  }
  if (cudaMemcpy(h->query_dev, query, query_needed * sizeof(float), cudaMemcpyDefault) != cudaSuccess) {
    rc = 13;
    goto done;
  }

  int64_t qs[2] = {nq, h->dim};
  int64_t nsh[2] = {nq, topk};
  int64_t qstrides[2] = {0, 0};
  int64_t nstrides[2] = {0, 0};
  DLManagedTensor qt, nt, dt;
  sk_init_dl_tensor(&qt, h->query_dev, nq, h->dim, kDLCUDA, 32, kDLFloat, qs, qstrides);
  sk_init_dl_tensor(&nt, h->ids_dev, nq, topk, kDLCUDA, 32, kDLUInt, nsh, nstrides);
  sk_init_dl_tensor(&dt, h->dist_dev, nq, topk, kDLCUDA, 32, kDLFloat, nsh, nstrides);

  if (required_itopk < 512) { required_itopk = 512; }
  if (h->sp->itopk_size < required_itopk) { h->sp->itopk_size = required_itopk; }
  h->sp->search_width = 8;
  cuvsFilter filter;
  filter.type = NO_FILTER;
  filter.addr = (uintptr_t)0;
  if (cuvsCagraSearch(h->res, h->sp, h->idx, &qt, &nt, &dt, filter) != CUVS_SUCCESS) {
    rc = 15;
    goto done;
  }
  if (cudaMemcpy(out_ids, h->ids_dev, output_needed * sizeof(unsigned int),
                 cudaMemcpyDefault) != cudaSuccess) {
    rc = 16;
    goto done;
  }
  if (out_dist != NULL &&
      cudaMemcpy(out_dist, h->dist_dev, output_needed * sizeof(float),
                 cudaMemcpyDefault) != cudaSuccess) {
    rc = 17;
    goto done;
  }

done:
  if (rc != 0) { fprintf(stderr, "[seekdb_cuvs] search rc=%d: %s\n", rc, cuvsGetLastErrorText()); }
  pthread_mutex_unlock(&h->mu);
  return rc;
}

void seekdb_cuvs_free(void* handle)
{
  sk_cuvs_index* h = (sk_cuvs_index*)handle;
  if (!h) { return; }
  if (h->query_dev) { cuvsRMMFree(h->res, h->query_dev, h->query_capacity * sizeof(float)); }
  if (h->ids_dev) { cuvsRMMFree(h->res, h->ids_dev, h->ids_capacity * sizeof(unsigned int)); }
  if (h->dist_dev) { cuvsRMMFree(h->res, h->dist_dev, h->dist_capacity * sizeof(float)); }
  if (h->sp) { cuvsCagraSearchParamsDestroy(h->sp); }
  if (h->idx) { cuvsCagraIndexDestroy(h->idx); }
  if (h->ip) { cuvsCagraIndexParamsDestroy(h->ip); }
  cuvsResourcesDestroy(h->res);
  pthread_mutex_destroy(&h->mu);
  free(h);
}

int seekdb_cuvs_serialize(void* handle, const char* filename)
{
  sk_cuvs_index* index = (sk_cuvs_index*)handle;
  if (!index || !filename) { return 1; }
  /* The outer seekdb snapshot already stores the FP32 vectors and external IDs.
   * Omitting the duplicate dataset keeps the temporary graph file graph-only. */
  if (cuvsCagraSerialize(index->res, filename, index->idx, false) != CUVS_SUCCESS) {
    fprintf(stderr, "[seekdb_cuvs] serialize failed: %s\n", cuvsGetLastErrorText());
    return 1;
  }
  return 0;
}

void* seekdb_cuvs_deserialize(const char* filename, const float* vectors, long n, long dim)
{
  if (!filename || !vectors || n <= 0 || dim <= 0) { return NULL; }
  sk_cuvs_index* index = (sk_cuvs_index*)calloc(1, sizeof(*index));
  if (!index) { return NULL; }
  if (pthread_mutex_init(&index->mu, NULL) != 0) { free(index); return NULL; }
  if (cuvsResourcesCreate(&index->res) != CUVS_SUCCESS) { goto fail_mu; }

  cuvsCagraIndex_t graph_index = NULL;
  if (cuvsCagraIndexCreate(&graph_index) != CUVS_SUCCESS) { goto fail_res; }
  if (cuvsCagraDeserialize(index->res, filename, graph_index) != CUVS_SUCCESS) { goto fail_graph; }
  if (cuvsCagraIndexCreate(&index->idx) != CUVS_SUCCESS) { goto fail_graph; }

  DLManagedTensor graph_tensor;
  memset(&graph_tensor, 0, sizeof(graph_tensor));
  if (cuvsCagraIndexGetGraph(graph_index, &graph_tensor) != CUVS_SUCCESS ||
      graph_tensor.dl_tensor.shape == NULL ||
      graph_tensor.dl_tensor.shape[0] != n ||
      graph_tensor.dl_tensor.shape[1] <= 0) {
    goto fail_idx;
  }
  const size_t graph_cols = (size_t)graph_tensor.dl_tensor.shape[1];
  if ((size_t)n > ((size_t)-1) / graph_cols / sizeof(unsigned int)) { goto fail_idx; }
  const size_t graph_bytes = (size_t)n * graph_cols * sizeof(unsigned int);
  void* graph_host = malloc(graph_bytes);
  if (graph_host == NULL ||
      cudaMemcpy(graph_host, graph_tensor.dl_tensor.data, graph_bytes, cudaMemcpyDefault) != cudaSuccess) {
    free(graph_host);
    goto fail_idx;
  }

  int64_t graph_shape[2] = {n, (int64_t)graph_cols};
  int64_t graph_strides[2] = {0, 0};
  DLManagedTensor host_graph;
  sk_init_dl_tensor(&host_graph, graph_host, n, (long)graph_cols,
                    kDLCPU, 32, kDLUInt, graph_shape, graph_strides);
  int64_t ds[2] = {n, dim};
  int64_t strides[2] = {0, 0};
  DLManagedTensor dset;
  sk_init_dl_tensor(&dset, (void*)vectors, n, dim, kDLCPU, 32, kDLFloat, ds, strides);
  const int from_args_rc =
      cuvsCagraIndexFromArgs(index->res, L2Expanded, &host_graph, &dset, index->idx);
  free(graph_host);
  if (from_args_rc != CUVS_SUCCESS) { goto fail_idx; }

  {
    int64_t actual_n = 0;
    int64_t actual_dim = 0;
    if (cuvsCagraIndexGetSize(index->idx, &actual_n) != CUVS_SUCCESS ||
        cuvsCagraIndexGetDims(index->idx, &actual_dim) != CUVS_SUCCESS ||
        actual_n != n || actual_dim != dim) {
      fprintf(stderr, "[seekdb_cuvs] snapshot shape mismatch: %ld x %ld, expected %ld x %ld\n",
              (long)actual_n, (long)actual_dim, n, dim);
      goto fail_idx;
    }
  }
  if (sk_create_search_params(index) != 0) { goto fail_idx; }
  cuvsCagraIndexDestroy(graph_index);
  index->n = n;
  index->dim = dim;
  return index;

fail_idx:
  cuvsCagraIndexDestroy(index->idx);
  index->idx = NULL;
fail_graph:
  cuvsCagraIndexDestroy(graph_index);
fail_res:
  cuvsResourcesDestroy(index->res);
fail_mu:
  fprintf(stderr, "[seekdb_cuvs] deserialize failed: %s\n", cuvsGetLastErrorText());
  pthread_mutex_destroy(&index->mu);
  free(index);
  return NULL;
}

int seekdb_cuvs_cagra_knn(const float* base, long n, long dim,
                          const float* query, long nq, long topk,
                          unsigned int* out_ids)
{
  void* h = seekdb_cuvs_build(base, n, dim);
  if (!h) { return 10; }
  int rc = seekdb_cuvs_search(h, query, nq, topk, out_ids, NULL);
  seekdb_cuvs_free(h);
  return rc;
}
