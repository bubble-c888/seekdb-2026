"""Vector helpers: literal parsing, L2 distance, top-k selection and k-means.

Vectors are stored as ``float32`` arrays.  Distances are reported as ``float64``
so that the printed value is reproducible, while the accumulated squared
distance is computed in ``float64`` as well to stay inside the ``1e-5``
tolerance that the task package demands for distance assertions.
"""

from __future__ import annotations

import math
import re
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..errors import DataError, DimensionMismatchError

FLOAT32_MAX = float(np.finfo(np.float32).max)

_NUMBER_RE = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")


def parse_vector_literal(text: str, dim: Optional[int] = None) -> np.ndarray:
    """Parse ``'[0, 2]'`` into a ``float32`` array.

    Rejects malformed input, a wrong number of dimensions, NaN, Infinity and
    values that cannot be represented as a finite ``float32``.
    """
    if not isinstance(text, str):
        raise DataError("vector literal must be a string")
    stripped = text.strip()
    if not (stripped.startswith("[") and stripped.endswith("]")):
        raise DataError(f"invalid vector literal: {text}")
    body = stripped[1:-1].strip()
    values: List[float] = []
    if body:
        for part in body.split(","):
            part = part.strip()
            if not part:
                raise DataError(f"invalid vector element: {part!r}")
            try:
                value = float(part)
            except ValueError:
                raise DataError(f"invalid vector element: {part!r}") from None
            values.append(value)
    if dim is not None and len(values) != dim:
        raise DimensionMismatchError(
            f"vector dimension mismatch: expected {dim}, got {len(values)}"
        )
    if not values:
        raise DataError("vector literal must not be empty")
    for value in values:
        if math.isnan(value) or math.isinf(value):
            raise DataError("vector values must be finite")
        if abs(value) > FLOAT32_MAX:
            raise DataError("vector value overflows float32")
    array = np.asarray(values, dtype=np.float32)
    if not np.all(np.isfinite(array)):
        raise DataError("vector value overflows float32")
    return array


def format_vector(vector: np.ndarray) -> List[float]:
    """Render a stored vector as plain Python floats."""
    return [float(value) for value in np.asarray(vector, dtype=np.float32).tolist()]


def l2_distances(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Euclidean distance between ``query`` and every row of ``matrix``."""
    if matrix.size == 0:
        return np.zeros(0, dtype=np.float64)
    diff = matrix.astype(np.float32, copy=False) - query.astype(np.float32, copy=False)
    squared = np.einsum("ij,ij->i", diff, diff, dtype=np.float64)
    return np.sqrt(squared)


def l2_distance(vector: np.ndarray, query: np.ndarray) -> float:
    diff = vector.astype(np.float32, copy=False) - query.astype(np.float32, copy=False)
    return float(math.sqrt(float(np.einsum("i,i->i", diff, diff, dtype=np.float64))))


def top_k_indices(distances: np.ndarray, k: int) -> np.ndarray:
    """Indices of the ``k`` smallest distances, sorted by (distance, row order)."""
    if k <= 0 or distances.size == 0:
        return np.zeros(0, dtype=np.int64)
    k = min(k, distances.size)
    part = np.argpartition(distances, k - 1)[:k]
    return part[np.argsort(distances[part], kind="stable")]


def kmeans(
    vectors: np.ndarray,
    k: int,
    iterations: int = 12,
    seed: int = 20261109,
) -> np.ndarray:
    """A small deterministic k-means used to build the coarse quantizer.

    The implementation is intentionally simple (k-means++ seeding followed by a
    few Lloyd iterations); the resulting index is an *approximate* index, the
    exact scan always stays available.
    """
    n = vectors.shape[0]
    if n == 0:
        return np.zeros((0, vectors.shape[1] if vectors.ndim == 2 else 0), dtype=np.float32)
    k = max(1, min(k, n))
    rng = np.random.default_rng(seed)
    # k-means++ seeding
    centroids = np.empty((k, vectors.shape[1]), dtype=np.float32)
    first = int(rng.integers(0, n))
    centroids[0] = vectors[first]
    closest = l2_distances(vectors, centroids[0]) ** 2
    for index in range(1, k):
        total = float(closest.sum())
        if total <= 0.0:
            centroids[index] = vectors[int(rng.integers(0, n))]
        else:
            probabilities = closest / total
            pick = int(np.searchsorted(np.cumsum(probabilities), rng.random()))
            pick = min(pick, n - 1)
            centroids[index] = vectors[pick]
        closest = np.minimum(closest, l2_distances(vectors, centroids[index]) ** 2)
    for _ in range(iterations):
        matrix = np.stack(
            [l2_distances(vectors, centroid) for centroid in centroids], axis=1
        )
        assignment = np.argmin(matrix, axis=1)
        for index in range(k):
            members = vectors[assignment == index]
            if members.shape[0]:
                centroids[index] = members.mean(axis=0).astype(np.float32)
    return centroids


def nearest_centroid(centroids: np.ndarray, vector: np.ndarray) -> int:
    if centroids.shape[0] == 0:
        return -1
    distances = l2_distances(centroids, vector)
    return int(np.argmin(distances))


def merge_top_k(
    best: List[Tuple[float, int, int]],
    distances: np.ndarray,
    pks: Sequence[int],
    slots: Sequence[int],
    k: int,
) -> None:
    """Merge a chunk of candidates into the running top-k list.

    ``best`` holds ``(distance, primary key, slot)`` triples ordered by
    distance; ties are broken by the primary key, exactly like the task
    statement requires.
    """
    if k <= 0 or distances.size == 0:
        return
    order = np.argsort(distances, kind="stable")
    for index in order:
        distance = float(distances[index])
        candidate = (distance, int(pks[index]), int(slots[index]))
        if len(best) < k:
            best.append(candidate)
            best.sort(key=lambda item: (item[0], item[1]))
        elif (candidate[0], candidate[1]) < (best[-1][0], best[-1][1]):
            best[-1] = candidate
            best.sort(key=lambda item: (item[0], item[1]))
