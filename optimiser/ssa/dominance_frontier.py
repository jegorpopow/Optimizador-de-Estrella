from optimiser.ssa.tree import Tree
from optimiser.ir.expressions import ControlFlow, BlockId


def dominance_frontiers(cfg: ControlFlow, dominators_tree: Tree) -> dict[BlockId, set[BlockId]]:
    blocks = cfg.named_blocks()
    children = dominators_tree.children()
    dominance_fronier = {b: set() for b in blocks}

    def go(block: BlockId):
        for successor in blocks[block].terminator.successors():
            if dominators_tree.parent[successor] != block:
                dominance_fronier[block].add(successor)
        for child in children[block]:
            go(child)
            for candidate in dominance_fronier[child]:
                if not dominators_tree.is_proper_ancestor(candidate, block):
                    dominance_fronier[block].add(candidate)

    go(dominators_tree.root())
    return dominance_fronier
