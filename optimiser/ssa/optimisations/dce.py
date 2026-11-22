from copy import deepcopy

from optimiser.ir.expressions import (
    Block,
    Callable,
    ControlFlow,
    Instructions,
    Location,
    Phi,
    Terminators,
    Usage,
)


class DCE:
    NAME = "DCE"

    def __init__(self, function: Callable):
        if function.body.form != "ssa":
            raise ValueError(f"DCE with called on (non-SSA) {function.body.form} form")
        self.function = deepcopy(function)
        self.vars = set(self.function.params) | self.function.body.all_bindings()

    def optimise(self) -> Callable:
        blocks = self.function.body.named_blocks()
        defs = self.function.body.all_defs_ssa({var.name for var in self.vars})

        alive: set[Usage] = set()
        roots: set[Usage] = set()

        for b in blocks:
            for i, instr in enumerate(blocks[b].instructions):
                if not Instructions.is_pure(instr):
                    roots.add(Usage(b, Location.Instruction(i)))
            roots.add(Usage(b, Location.Terminator()))

        while roots:
            root = roots.pop()
            if not root in alive:
                alive.add(root)
                obj = blocks[root.block].at(root.place)

                if Phi.is_phi(obj):
                    used = obj.used_vars()
                elif Instructions.is_instruction(obj):
                    used = Instructions.used_vars(obj)
                elif Terminators.is_terminator(obj):
                    used = Terminators.used_vars(obj)
                else:
                    raise ValueError(f"Unknown construction {obj}")
                for var in used:
                    if defs[var] is not None:
                        roots.add(defs[var])

        new_blocks = [
            Block(
                name=b,
                phi_nodes=[
                    phi
                    for i, phi in enumerate(blocks[b].phi_nodes)
                    if Usage(b, Location.Phi(i)) in alive
                ],
                instructions=[
                    instr
                    for i, instr in enumerate(blocks[b].instructions)
                    if Usage(b, Location.Instruction(i)) in alive
                ],
                terminator=blocks[b].terminator,
            )
            for b in blocks
        ]

        return Callable(
            kind=self.function.kind,
            name=self.function.name,
            params=self.function.params,
            result=self.function.result,
            body=ControlFlow(
                form="ssa",
                locals=self.function.body.locals,
                entry=self.function.body.entry,
                blocks=new_blocks,
            ),
        )
