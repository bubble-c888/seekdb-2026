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
        radii: Optional[Dict[str, float]] = None,
    ):
        self.table = table
        self.name = name
        self.column = column
        self.dim = dim
        self.clusters = clusters
        self.nprobe = max(1, min(nprobe, max(1, clusters)))
        self.lists: Dict[str, int] = dict(lists or {})
        #: The radius of every cluster: the greatest distance of a member to
        #: the centroid.  Together with the distance of the query to the
        #: centroid it bounds the distance of every row of the cluster.
        self.radii: Dict[str, float] = dict(radii or {})
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
            "radii": {key: float(value) for key, value in self.radii.items()},
        }

    # -- building -------------------------------------------------------
    def build(self, writes: WriteSet, sample_limit: int = 50000) -> None:
        """(Re)build the coarse quantizer and the inverted lists.

        The work runs block by block: a table that holds more vectors than the
        memory budget allows can be indexed, because neither the whole column
        nor the distances of the whole table are in the memory at once.
        """
        table = self.table
        column = table.column(self.column)
        alive = table.read_alive(0, table.n_slots)
        live_slots = np.nonzero(alive)[0]
        if live_slots.size == 0:
            self.clusters = 0
            self.lists = {}
            return
        # Only the sample of the quantizer is read into the memory.
        if live_slots.size > sample_limit:
            step = max(1, live_slots.size // sample_limit)
            sample_slots = live_slots[::step][:sample_limit]
        else:
            sample_slots = live_slots
        sample = self._gather_vectors(column, sample_slots)
        clusters = max(1, min(self.clusters or 64, sample.shape[0]))
        centroids = kmeans(sample.astype(np.float32), clusters)
        self.centroids = centroids
        self.clusters = int(centroids.shape[0])
        self.nprobe = max(1, min(self.nprobe, self.clusters))
        writes.add(self.centroid_key, 0, centroids.astype(np.float32).tobytes())
        members: Dict[int, List[int]] = {
            cluster: [] for cluster in range(self.clusters)
        }
        radii = [0.0] * self.clusters
        block = 8192
        for start in range(0, live_slots.size, block):
            slots = live_slots[start : start + block]
            vectors = self._gather_vectors(column, slots)
            distances = np.stack(
                [l2_distances(vectors, centroid) for centroid in centroids], axis=1
            )
            assignment = np.argmin(distances, axis=1)
            for position, cluster in enumerate(assignment.tolist()):
                members[cluster].append(int(slots[position]))
            for cluster in range(self.clusters):
                selected = assignment == cluster
                if not np.any(selected):
                    continue
                radius = float(np.max(distances[selected, cluster]))
                if radius > radii[cluster]:
                    radii[cluster] = radius
        self.lists = {}
        self.radii = {}
        for cluster in range(self.clusters):
            rows = members[cluster]
            if rows:
                payload = np.asarray(rows, dtype=np.int32).tobytes()
                writes.add(self.list_key(cluster), 0, payload)
                self.lists[str(cluster)] = len(rows)
                self.radii[str(cluster)] = radii[cluster]
            else:
                self.lists[str(cluster)] = 0
                self.radii[str(cluster)] = 0.0

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
        # The radius has to cover the new row as well.
        radius = float(l2_distances(np.asarray([vector], dtype=np.float32), centroids[cluster])[0])
        if radius > self.radii.get(str(cluster), 0.0):
            self.radii[str(cluster)] = radius

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
        for _distance, _radius, cluster_slots in self.iter_cluster_slots(query):
            for slot in cluster_slots:
                if slot not in seen:
                    seen.add(slot)
                    slots.append(int(slot))
        return slots

    def iter_cluster_slots(self, query: np.ndarray):
        """Yield ``(cluster distance, radius, slots)`` for every cluster.

        The clusters are ordered by the distance of their centroid to the query.
        The caller can stop early once the bound of a cluster proves that it
        cannot hold a better row, which keeps the approximate search cheap.
        """
        centroids = self.read_centroids()
        if centroids is None or centroids.shape[0] == 0:
            return
        distances = l2_distances(centroids, query.astype(np.float32))
        for cluster in np.argsort(distances, kind="stable"):
            key = self.list_key(int(cluster))
            raw = self.table.store.read_all(key)
            count = int(self.lists.get(str(int(cluster)), 0))
            radius = float(self.radii.get(str(int(cluster)), 0.0))
            if not raw or count <= 0:
                yield float(distances[cluster]), radius, []
                continue
            values = np.frombuffer(raw, dtype=np.int32)[:count]
            yield float(distances[cluster]), radius, values.tolist()

    # -- helpers --------------------------------------------------------
    def gather_vectors(self, column, slots: np.ndarray) -> np.ndarray:
        """Read the vectors of the given slots, block by block."""
        return self._gather_vectors(column, slots)

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
