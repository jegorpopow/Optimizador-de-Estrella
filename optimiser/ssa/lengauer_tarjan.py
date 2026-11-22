from optimiser.ir.expressions import ControlFlow, BlockId
from optimiser.ssa.tree import Tree


def lengauer_tarjan(cfg: ControlFlow) -> Tree[BlockId]:
    blocks = cfg.named_blocks()
    dfnum = {b: None for b in blocks}
    parent = {b: None for b in blocks}
    predecessors = {b: [] for b in blocks}
    preorder = []

    def on_enter(p, v):
        parent[v] = p
        preorder.append(v)
        dfnum[v] = len(preorder) - 1

    def on_edge(p, v):
        predecessors[v].append(p)

    cfg.dfs(on_enter=on_enter, on_edge=on_edge)
    semidominator = {b: None for b in blocks}

    class LinkEval[T]:
        def __init__(self):
            self.ancestor = {v: None for v in blocks}
            self.label = {v: v for v in blocks}

        def link(self, p: T, v: T):
            self.ancestor[v] = p

        def eval(self, v: T):
            a = self.ancestor[v]
            if a != None and self.ancestor[a] is not None:
                b = self.eval(a)
                self.ancestor[v] = self.ancestor[a]
                if dfnum[semidominator[b]] < dfnum[semidominator[self.label[v]]]:
                    self.label[v] = b
            return self.label[v]

    link_eval = LinkEval()
    bucket = {b: set() for b in blocks}

    immediate_dominator = {b: None for b in blocks}
    codominated = {b: None for b in blocks}

    for n in preorder[:0:-1]:
        p = parent[n]
        semidominator[n] = p
        for v in predecessors[n]:
            if dfnum[v] <= dfnum[n] and dfnum[v] < dfnum[semidominator[n]]:
                semidominator[n] = v
            elif (
                dfnum[v] > dfnum[n]
                and dfnum[candidate := semidominator[link_eval.eval(v)]] < dfnum[semidominator[n]]
            ):
                semidominator[n] = candidate
        bucket[semidominator[n]].add(n)
        link_eval.link(p, n)

        for v in bucket[p]:
            y = link_eval.eval(v)
            if semidominator[y] == semidominator[v]:
                immediate_dominator[v] = p
            else:
                codominated[v] = y
        bucket[p].clear()

    for n in preorder[1:]:
        if codominated[n] != None:
            immediate_dominator[n] = immediate_dominator[codominated[n]]

    return Tree[BlockId](immediate_dominator)
