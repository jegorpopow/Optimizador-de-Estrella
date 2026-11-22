from optimiser.ir.expressions import Callable, ControlFlow


class UnreachableBlocksElimination:
    NAME = "UnreachableBlocksElimination"

    def __init__(self, function: Callable):
        self.function = function

    def optimise(self):
        predecessors = self.function.body.predecessors()

        return Callable(
            kind=self.function.kind,
            name=self.function.name,
            params=self.function.params,
            result=self.function.result,
            body=ControlFlow(
                form=self.function.body.form,
                entry=self.function.body.entry,
                locals=self.function.body.locals,
                blocks=[
                    block
                    for block in self.function.body.blocks
                    if block.name == self.function.body.entry or len(predecessors[block.name]) > 0
                ],
            ),
        )
