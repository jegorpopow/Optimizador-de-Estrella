from copy import copy, deepcopy
from dataclasses import dataclass

from optimiser.ir.expressions import (
    Block,
    BlockId,
    ControlFlow,
    Callable,
    Binding,
    Edge,
    Operand,
    Terminators,
    UnOp,
    Variable,
    Operands,
    Instructions,
    Phi,
    StellaIdent,
)
from optimiser.ssa.tree import Tree
from optimiser.ssa.lengauer_tarjan import lengauer_tarjan
from optimiser.ssa.dominance_frontier import dominance_frontiers
from optimiser.analysis.liveness import LivenessAnalysis


class SSABuilder:
    def __init__(self, function: Callable):
        self.function = function
        self.cfg = self.function.body
        self.blocks = self.cfg.named_blocks()
        self.bindings = set(self.function.params) | self.cfg.all_bindings()
        self.predecessors = self.cfg.predecessors()
        self.dominator = lengauer_tarjan(self.cfg)
        self.dominance_frontier = dominance_frontiers(self.cfg, self.dominator)
        self.children = self.dominator.children()
        self.liveness_on_enter, _ = LivenessAnalysis(function.body).analyse()

    @staticmethod
    def ssa_var(var: StellaIdent, index: int) -> StellaIdent:
        return f"ssa_{index}_{var}"

    def phi_sites(self) -> dict[BlockId, set[Binding]]:
        defined = {b.name: set(b.defined_vars()) for b in self.cfg.blocks}

        defsites: dict[Binding, set[BlockId]] = {var: set() for var in self.bindings}
        phies: dict[Binding, set[BlockId]] = {b: set() for b in self.bindings}

        for b in self.blocks:
            for v in defined[b]:
                defsites[v].add(b)

        for var in self.bindings:
            w = copy(defsites[var])
            while w:
                n = w.pop()
                for y in self.dominance_frontier[n]:
                    if not y in phies[var]:
                        phies[var].add(y)
                        if not var in defined[y]:
                            w.add(y)

        result: dict[BlockId, set[Binding]] = {b: set() for b in self.blocks}
        for var in phies:
            for b in phies[var]:
                if var.name in self.liveness_on_enter[b]:
                    result[b].add(var)
        return result

    def fresh_ssa_block(self, b: Block, vars: set[Binding]) -> Block:
        return Block(
            name=b.name,
            phi_nodes=[
                Phi(
                    op="phi",
                    result=var,
                    incoming={p: Variable(var.name) for p in self.predecessors[b.name]},
                )
                for var in vars
            ],
            instructions=deepcopy(b.instructions),
            terminator=deepcopy(b.terminator),
        )

    def ssa(self) -> Callable:
        params_names = set([p.name for p in self.function.params])
        count = {v.name: 0 for v in self.bindings}
        stack = {v.name: [v.name] if v.name in params_names else [None] for v in self.bindings}

        def get_current_name(name: StellaIdent):
            if len(stack[name]) == 0:
                raise RuntimeError(f"Variable {name} was read before initialisation")
            return stack[name][-1]

        phi_sites = self.phi_sites()
        blocks = {b.name: self.fresh_ssa_block(b, phi_sites[b.name]) for b in self.cfg.blocks}

        def fresh_name(name: StellaIdent):
            count[name] += 1
            stack[name].append(self.ssa_var(name, count[name]))
            return count[name]

        def rename(b: BlockId):
            depths = {v: len(stack[v]) for v in stack}
            for phi in blocks[b].phi_nodes:
                fresh_name(phi.result.name)
                phi.result = phi.result.rename(lambda name: stack[name][-1])
            for i, instruction in enumerate(blocks[b].instructions):
                instruction = Instructions.map_args(
                    instruction, lambda op: Operands.rename_vars(op, get_current_name)
                )
                for var in Instructions.defined_vars(instruction):
                    fresh_name(var.name)
                instruction = Instructions.rename_results(instruction, get_current_name)
                blocks[b].instructions[i] = instruction

            blocks[b].terminator = Terminators.map_operands(
                blocks[b].terminator, lambda op: Operands.rename_vars(op, get_current_name)
            )

            for succ in blocks[b].terminator.successors():
                for phi in blocks[succ].phi_nodes:
                    phi.incoming[b] = Operands.rename_vars(phi.incoming[b], get_current_name)

            for child in self.children[b]:
                rename(child)

            for v in stack:
                del stack[v][depths[v] :]

        rename(self.cfg.entry)

        return Callable(
            kind=self.function.kind,
            name=self.function.name,
            params=self.function.params,
            result=self.function.result,
            body=ControlFlow("ssa", [], self.cfg.entry, [blocks[b] for b in blocks]),
        )


class SSALeave:
    def __init__(self, function: Callable):
        if function.body.form != "ssa":
            raise ValueError(f"Leave from SSA invoked on function in {function.body.form} form")
        self.function = deepcopy(function)
        self.params_set = set(self.function.params)
        self.max_block_name_len = max(len(b.name) for b in self.function.body.blocks)
        self.fresh_counter = 0

    def fresh_block_id(self) -> BlockId:
        self.fresh_counter += 1
        return "A" * self.max_block_name_len + f"_{self.fresh_counter}"

    def eliminate_critical(self):
        blocks = self.function.body.named_blocks()
        predecessors = self.function.body.predecessors()
        successors = self.function.body.successors()

        def is_critical(edge: Edge) -> bool:
            return len(successors[edge.source]) > 1 and len(predecessors[edge.destination]) > 1

        critical_edges = [
            Edge(b, d) for b in blocks for d in successors[b] if is_critical(Edge(b, d))
        ]

        for edge in critical_edges:
            self.function.body.split_edge(edge, self.fresh_block_id())

    def fresh_var(self, name: StellaIdent) -> StellaIdent:
        self.fresh_counter += 1
        return f"cfg_var_{self.fresh_counter}_{name}"

    def cfg(self) -> Callable:
        self.eliminate_critical()

        blocks = self.function.body.named_blocks()

        exit_copies: dict[BlockId, list[tuple[Binding, Operand]]] = {b: [] for b in blocks}
        for b in blocks:
            for phi in blocks[b].phi_nodes:
                for pred in phi.incoming:
                    exit_copies[pred].append((phi.result, phi.incoming[pred]))

        def compile_simultaneous_copies(
            copies: list[tuple[Binding, Operand]],
        ) -> list[Instructions]:
            tmps = [self.fresh_var(var.name) for var, _ in copies]
            result = []
            for tmp, (var, operand) in zip(tmps, copies):
                result.append(
                    UnOp(
                        op="copy",
                        arg=operand,
                        result=Binding(name=tmp, type=var.type),
                    )
                )
            for tmp, (var, operand) in zip(tmps, copies):
                result.append(UnOp(op="copy", arg=Variable(name=tmp), result=var))

            return result

        new_blocks = [
            Block(
                name=b,
                phi_nodes=[],
                instructions=blocks[b].instructions + compile_simultaneous_copies(exit_copies[b]),
                terminator=blocks[b].terminator,
            )
            for b in blocks
        ]

        locals = list(
            {
                result
                for block in new_blocks
                for instr in block.instructions
                for result in Instructions.defined_vars(instr)
                if not result in self.params_set
            }
        )

        return Callable(
            kind=self.function.kind,
            name=self.function.name,
            params=self.function.params,
            result=self.function.result,
            body=ControlFlow(
                form="cfg", locals=locals, entry=self.function.body.entry, blocks=new_blocks
            ),
        )
