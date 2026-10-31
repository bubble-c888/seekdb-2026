/* FP16 graph construction with an FP32 CAGRA index for search.
 *
 * hipVS accepts host/device FP16 datasets for NN-Descent. Building the graph
 * from FP16 halves the graph-construction dataset traffic while the final
 * CAGRA index is reconstructed against the original FP32 vectors. This keeps
 * seekdb's external vector semantics and search distances in FP32.
 */
#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/cagra.h>
#include <dlpack/dlpack.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace {

void fill_tensor(DLManagedTensor* tensor,
                 void* data,
                 long rows,
                 long dim,
                 int bits,
                 DLDataTypeCode code,
                 int64_t* shape,
                 int64_t* strides)
{
  std::memset(tensor, 0, sizeof(*tensor));
  shape[0] = rows;
  shape[1] = dim;
  tensor->dl_tensor.data = data;
  tensor->dl_tensor.device.device_type = kDLCPU;
  tensor->dl_tensor.device.device_id = 0;
  tensor->dl_tensor.ndim = 2;
  tensor->dl_tensor.dtype.code = code;
  tensor->dl_tensor.dtype.bits = static_cast<uint8_t>(bits);
  tensor->dl_tensor.dtype.lanes = 1;
  tensor->dl_tensor.shape = shape;
  tensor->dl_tensor.strides = strides;
  tensor->dl_tensor.byte_offset = 0;
}

}  // namespace

extern "C" int seekdb_cuvs_build_fp16_float_graph(
    const float* base,
    long n,
    long dim,
    size_t graph_degree,
    size_t intermediate_graph_degree,
    size_t nn_descent_niter,
    cuvsResources_t res,
    cuvsCagraIndex_t* out_index)
{
  if (base == nullptr || n <= 0 || dim <= 0 || out_index == nullptr) {
    return 1;
  }
  *out_index = nullptr;

  const size_t count = static_cast<size_t>(n);
  const size_t width = static_cast<size_t>(dim);
  if (count > std::numeric_limits<size_t>::max() / width) { return 2; }
  std::vector<__half> half_vectors;
  try {
    half_vectors.resize(count * width);
  } catch (...) {
    return 3;
  }
  for (size_t i = 0; i < count * width; ++i) {
    half_vectors[i] = __float2half_rn(base[i]);
  }

  cuvsCagraIndexParams_t graph_params = nullptr;
  cuvsCagraIndex_t graph_index = nullptr;
  int rc = 0;
  if (cuvsCagraIndexParamsCreate(&graph_params) != CUVS_SUCCESS) {
    rc = 4;
    goto done;
  }
  graph_params->build_algo = NN_DESCENT;
  graph_params->graph_degree = graph_degree;
  graph_params->intermediate_graph_degree = intermediate_graph_degree;
  graph_params->nn_descent_niter = nn_descent_niter;
  if (cuvsCagraIndexCreate(&graph_index) != CUVS_SUCCESS) {
    rc = 5;
    goto done;
  }
  {
    int64_t shape[2] = {n, dim};
    int64_t strides[2] = {0, 0};
    DLManagedTensor half_dataset;
    fill_tensor(&half_dataset, half_vectors.data(), n, dim, 16, kDLFloat, shape, strides);
    if (cuvsCagraBuild(res, graph_params, &half_dataset, graph_index) != CUVS_SUCCESS) {
      rc = 6;
      goto done;
    }
  }

  if (cuvsCagraIndexCreate(out_index) != CUVS_SUCCESS) {
    rc = 7;
    goto done;
  }
  {
    DLManagedTensor graph;
    std::memset(&graph, 0, sizeof(graph));
    if (cuvsCagraIndexGetGraph(graph_index, &graph) != CUVS_SUCCESS ||
        graph.dl_tensor.shape == nullptr ||
        graph.dl_tensor.shape[0] != n ||
        graph.dl_tensor.shape[1] <= 0) {
      rc = 8;
      goto done;
    }
    const size_t graph_cols = static_cast<size_t>(graph.dl_tensor.shape[1]);
    if (count > std::numeric_limits<size_t>::max() / graph_cols / sizeof(uint32_t)) {
      rc = 8;
      goto done;
    }
    std::vector<uint32_t> graph_host;
    try {
      graph_host.resize(count * graph_cols);
    } catch (...) {
      rc = 8;
      goto done;
    }
    const size_t graph_bytes = graph_host.size() * sizeof(uint32_t);
    if (hipMemcpy(graph_host.data(), graph.dl_tensor.data, graph_bytes,
                  hipMemcpyDeviceToHost) != hipSuccess) {
      rc = 8;
      goto done;
    }
    int64_t graph_shape[2] = {n, static_cast<int64_t>(graph_cols)};
    int64_t graph_strides[2] = {0, 0};
    DLManagedTensor host_graph;
    fill_tensor(&host_graph, graph_host.data(), n, static_cast<long>(graph_cols),
                32, kDLUInt, graph_shape, graph_strides);
    int64_t shape[2] = {n, dim};
    int64_t strides[2] = {0, 0};
    DLManagedTensor float_dataset;
    fill_tensor(&float_dataset, const_cast<float*>(base), n, dim, 32, kDLFloat, shape, strides);
    if (cuvsCagraIndexFromArgs(res, L2Expanded, &host_graph, &float_dataset, *out_index) != CUVS_SUCCESS) {
      rc = 9;
      goto done;
    }
  }

done:
  if (rc != 0) {
    std::fprintf(stderr, "[seekdb_cuvs] FP16 graph build failed rc=%d: %s\n",
                 rc, cuvsGetLastErrorText());
    if (*out_index != nullptr) {
      cuvsCagraIndexDestroy(*out_index);
      *out_index = nullptr;
    }
  }
  if (graph_index != nullptr) {
    cuvsCagraIndexDestroy(graph_index);
  }
  if (graph_params != nullptr) {
    cuvsCagraIndexParamsDestroy(graph_params);
  }
  return rc;
}
