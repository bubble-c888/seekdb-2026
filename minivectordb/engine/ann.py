"""Approximate vector index (IVF style) implemented on top of the table files.

The index is optional: the exact scan is always available and always correct.
The index is maintained incrementally - a row that is inserted, updated or
deleted is reflected in the search results immediately - and it is rebuilt from
the persisted rows whenever the database starts.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .storage import FileStore, WriteSet
from .table import Table
from .vector import kmeans, l2_distances, nearest_centroid


class VectorIndex:
    """An IVF index over one vector column."""

    def __init__(
        self,
        table: Table,
        name: str,
        column: str,
        dim: int,
        clusters: int,
        nprobe: int,
        lists: Optional[Dict[str, int]] = None,
    ):
        self.table = table
        self.name = name
        self.column = column
        self.dim = dim
        self.clusters = clusters
        self.nprobe = max(1, min(nprobe, max(1, clusters)))
        self.lists: Dict[str, int] = dict(lists or {})
        self.centroids: Optional[np.ndarray] = None

    # -- file keys ------------------------------------------------------
    @property
    def centroid_key(self) -> str:
        return f"t/{self.table.name}/idx_{self.name}.cent"

    def list_key(self, cluster: int) -> str:
        return f"t/{self.table.name}/idx_{self.name}_list_{cluster}.dat"

    # -- metadata -------------------------------------------------------
    def to_meta(self) -> Dict[str, Any]:
        return {
            "kind": "ivf",
            "column": self.column,
            "dim": self.dim,
            "clusters": self.clusters,
            "nprobe": self.nprobe,
            "lists": self.lists,
        }

    # -- building -------------------------------------------------------
    def build(self, writes: WriteSet, sample_limit: int = 50000) -> None:
        """(Re)build the coarse quantizer and the inverted lists."""
        table = self.table
        column = table.column(self.column)
        alive = table.read_alive(0, table.n_slots)
        live_slots = np.nonzero(alive)[0]
        if live_slots.size == 0:
            self.clusters = 0
            self.lists = {}
            return
        vectors = self._gather_vectors(column, live_slots)
        sample = vectors
        if vectors.shape[0] > sample_limit:
            step = max(1, vectors.shape[0] // sample_limit)
            sample = vectors[::step]
        clusters = max(1, min(self.clusters or 64, sample.shape[0]))
        centroids = kmeans(sample.astype(np.float32), clusters)
        self.centroids = centroids
        self.clusters = int(centroids.shape[0])
        self.nprobe = max(1, min(self.nprobe, self.clusters))
        writes.add(self.centroid_key, 0, centroids.astype(np.float32).tobytes())
        assignment = np.argmin(
            np.stack(
                [l2_distances(vectors, centroid) for centroid in centroids], axis=1
            ),
            axis=1,
        )
        self.lists = {}
        for cluster in range(self.clusters):
            members = live_slots[assignment == cluster]
            if members.size:
                payload = members.astype(np.int32).tobytes()
                writes.add(self.list_key(cluster), 0, payload)
                self.lists[str(cluster)] = int(members.size)
            else:
                self.lists[str(cluster)] = 0

    def assign(self, slot: int, vector: np.ndarray, writes: WriteSet) -> None:
        """Add a freshly inserted row to the closest inverted list."""
        centroids = self.read_centroids()
        if centroids is None or centroids.shape[0] == 0:
            return
        cluster = nearest_centroid(centroids, vector)
        if cluster < 0:
            return
        key = self.list_key(cluster)
        offset = writes.logical_size(key, self.table.store.size(key))
        writes.add(key, offset, np.asarray([slot], dtype=np.int32).tobytes())
        self.lists[str(cluster)] = self.lists.get(str(cluster), 0) + 1

    # -- reading --------------------------------------------------------
    def read_centroids(self) -> Optional[np.ndarray]:
        if self.centroids is not None:
            return self.centroids
        raw = self.table.store.read_all(self.centroid_key)
        if not raw:
            return None
        count = len(raw) // (4 * self.dim) if self.dim else 0
        if count == 0:
            return None
        return np.frombuffer(raw, dtype=np.float32, count=count * self.dim).reshape(
            count, self.dim
        )

    def candidate_slots(self, query: np.ndarray) -> List[int]:
        """Return the row slots that the approximate search has to consider."""
        slots: List[int] = []
        seen = set()
        for _distance, cluster_slots in self.iter_cluster_slots(query):
            for slot in cluster_slots:
                if slot not in seen:
                    seen.add(slot)
                    slots.append(int(slot))
        return slots

    def iter_cluster_slots(self, query: np.ndarray):
        """Yield ``(cluster distance, slots)`` for every cluster, closest first.

        The caller can stop early once it collected enough candidates, which
        keeps the approximate search cheap when a filter rejects most rows.
        """
        centroids = self.read_centroids()
        if centroids is None or centroids.shape[0] == 0:
            return
        distances = l2_distances(centroids, query.astype(np.float32))
        for cluster in np.argsort(distances, kind="stable"):
            key = self.list_key(int(cluster))
            raw = self.table.store.read_all(key)
            count = int(self.lists.get(str(int(cluster)), 0))
            if not raw or count <= 0:
                yield float(distances[cluster]), []
                continue
            values = np.frombuffer(raw, dtype=np.int32)[:count]
            yield float(distances[cluster]), values.tolist()

    # -- helpers --------------------------------------------------------
    def _gather_vectors(self, column, slots: np.ndarray) -> np.ndarray:
        dim = int(column.length or 0)
        if slots.size == 0:
            return np.zeros((0, dim), dtype=np.float32)
        result = np.zeros((slots.size, dim), dtype=np.float32)
        # Read in ascending order so that the file is accessed sequentially.
        order = np.argsort(slots)
        sorted_slots = slots[order]
        block = 4096
        for start in range(0, sorted_slots.size, block):
            chunk_slots = sorted_slots[start : start + block]
            first = int(chunk_slots[0])
            last = int(chunk_slots[-1])
            span = last - first + 1
            if span > 4 * chunk_slots.size + 64:
                # Sparse slots: read every row individually.
                for offset, slot in enumerate(chunk_slots.tolist()):
                    result[order[start + offset]] = self.table.read_vectors(
                        column, slot, 1
                    )[0]
            else:
                block_vectors = self.table.read_vectors(column, first, span)
                for offset, slot in enumerate(chunk_slots.tolist()):
                    result[order[start + offset]] = block_vectors[slot - first]
        return result
