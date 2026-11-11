import heapq
import math


class VectorIndex:
    """Vantage point tree with a bounded search and exact candidate distances."""

    def __init__(self, vectors):
        self.vectors = vectors
        self.root = self._build(list(vectors))

    def _build(self, keys):
        if len(keys) <= 24:
            return keys
        pivot = keys.pop(len(keys) // 2)
        vector = self.vectors[pivot]
        ordered = sorted((math.dist(vector, self.vectors[key]), key) for key in keys)
        middle = len(ordered) // 2
        return (
            pivot, ordered[middle][0],
            self._build([key for _, key in ordered[:middle]]),
            self._build([key for _, key in ordered[middle:]]),
        )

    def search(self, query, allowed, count):
        best = []
        remaining = max(4096, 32 * count, len(self.vectors) // 4)

        def distance(key):
            nonlocal remaining
            remaining -= 1
            value = math.dist(query, self.vectors[key])
            if key in allowed:
                entry = (-value, -key)
                if len(best) < count:
                    heapq.heappush(best, entry)
                elif entry > best[0]:
                    heapq.heapreplace(best, entry)
            return value

        def radius():
            return -best[0][0] if len(best) == count else math.inf

        def visit(node):
            if remaining <= 0:
                return
            if isinstance(node, list):
                for key in node[:remaining]:
                    if key in allowed:
                        distance(key)
                return
            key, split, inner, outer = node
            value = distance(key)
            if value <= split:
                visit(inner)
                if value + radius() >= split:
                    visit(outer)
            else:
                visit(outer)
                if value - radius() <= split:
                    visit(inner)

        visit(self.root)
        return [-key for _, key in best]
