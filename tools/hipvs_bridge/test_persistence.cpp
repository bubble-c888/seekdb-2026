#include "seekdb_cuvs_bridge.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static bool verify_search(void *index, const std::vector<float> &vectors, long topk)
{
  std::vector<unsigned int> ids(static_cast<size_t>(topk));
  std::vector<float> distances(static_cast<size_t>(topk));
  if (seekdb_cuvs_search(index, vectors.data(), 1, topk, ids.data(), distances.data()) != 0) {
    return false;
  }
  return ids[0] == 0 && std::fabs(distances[0]) <= 0.0001f;
}

int main(int argc, char **argv)
{
  if (argc != 3) return 2;
  const long count = 10000;
  const long dimension = 32;
  std::vector<float> vectors(count * dimension);
  for (long row = 0; row < count; ++row) {
    for (long column = 0; column < dimension; ++column) {
      vectors[row * dimension + column] = std::sin(float(row * 31 + column * 17));
    }
  }
  void *index = nullptr;
  if (std::strcmp(argv[1], "write") == 0) {
    index = seekdb_cuvs_build(vectors.data(), count, dimension);
    if (!index || seekdb_cuvs_serialize(index, argv[2]) != 0) return 3;
  } else {
    void *invalid = seekdb_cuvs_deserialize(argv[2], vectors.data(), count - 1, dimension);
    if (invalid) { seekdb_cuvs_free(invalid); return 6; }
    invalid = seekdb_cuvs_deserialize(argv[2], vectors.data(), count, dimension + 1);
    if (invalid) { seekdb_cuvs_free(invalid); return 7; }
    index = seekdb_cuvs_deserialize(argv[2], vectors.data(), count, dimension);
    if (!index) return 4;
  }
  if (!verify_search(index, vectors, 10) ||
      !verify_search(index, vectors, 100) ||
      !verify_search(index, vectors, 10)) {
    seekdb_cuvs_free(index);
    return 5;
  }
  seekdb_cuvs_free(index);
  std::printf("%s: n=%ld top1=0 reused-buffer=ok\n", argv[1], count);
  return 0;
}
