from dataclasses import dataclass


@dataclass
class Tree[T]:
    parent: dict[T, T | None]

    def children(self) -> dict[T, set[T]]:
        result = {b: set() for b in self.parent}
        for b in self.parent:
            if self.parent[b] is not None:
                result[self.parent[b]].add(b)
        return result

    def is_proper_ancestor(self, child: T, parent: T) -> bool:
        child = self.parent[child]
        while child is not None and child != parent:
            child = self.parent[child]
        return child == parent

    def is_ancestor(self, child: T, parent: T) -> bool:
        return child == parent or self.is_proper_ancestor(child, parent)

    def root(self) -> T:
        node = next(iter(self.parent))
        while self.parent[node] is not None:
            node = self.parent[node]
        return node
