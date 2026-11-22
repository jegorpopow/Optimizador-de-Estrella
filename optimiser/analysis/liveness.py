from optimiser.ir.expressions import ControlFlow, BlockId, Instructions, StellaIdent, Terminators


class LivenessAnalysis:
    def __init__(self, cfg: ControlFlow):
        if cfg.form != "cfg":
            raise ValueError(f"Liveness analysis invoked on {cfg.form} instead of cfg")
        self.cfg = cfg
        self.blocks = self.cfg.named_blocks()

    def uses(self, b: BlockId) -> set[StellaIdent]:
        block = self.blocks[b]

        defined_vars: set[StellaIdent] = set()
        result: set[StellaIdent] = set()

        for instruction in block.instructions:
            result |= set(Instructions.used_vars(instruction)) - defined_vars
            defined_vars |= set(Instructions.defined_vars(instruction))

        result |= set(Terminators.used_vars(block.terminator)) - defined_vars
        return result

    def defs(self, b: BlockId) -> set[StellaIdent]:
        return {var.name for var in self.blocks[b].defined_vars()}

    def analyse(self) -> tuple[dict[BlockId, set[StellaIdent]], dict[BlockId, set[StellaIdent]]]:
        postorder = []
        self.cfg.dfs(on_exit=lambda v: postorder.append(v))

        uses = {b: self.uses(b) for b in postorder}
        defs = {b: self.defs(b) for b in postorder}

        in_set = {b: set() for b in postorder}
        out_set = {b: set() for b in postorder}

        flag = True
        while flag:
            flag = False
            for b in postorder:
                old_out = out_set[b]
                out_set[b] = set().union(
                    *(in_set[p] for p in self.blocks[b].terminator.successors())
                )
                in_set[b] = uses[b] | (out_set[b] - defs[b])
                flag = flag or (old_out != out_set[b])

        return in_set, out_set
