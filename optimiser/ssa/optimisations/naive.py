from copy import deepcopy

from optimiser.ir.expressions import (
    Callable,
    Instruction,
    Instructions,
    Operand,
    Phi,
    StellaIdent,
    Terminators,
    UnOp,
    Variable,
)


class NaiveAliasPropagation:
    NAME = "NaiveAliasPropagation"

    def __init__(self, function: Callable):
        if function.body.form != "ssa":
            raise ValueError(f"Naive folding with called on (non-SSA) {function.body.form} form")
        self.function = deepcopy(function)

    @staticmethod
    def is_phi_trivial(phi: Phi) -> Operand | None:
        if len(phi_vals := set(phi.incoming.values())) == 1:
            return next(iter(phi_vals))
        else:
            return None

    @staticmethod
    def is_instruction_alias(instr: Instruction) -> Operand | None:
        match instr:
            case UnOp(op="copy", arg=Variable() as arg):
                return arg
            case _:
                return None

    @staticmethod
    def chase_aliases(aliases: dict[StellaIdent, Operand]) -> dict[StellaIdent, Operand]:
        result = {}

        def chase(name: StellaIdent, seen=frozenset()) -> Operand:
            if name in result:
                return result[name]
            if name in seen:
                raise ValueError(f"Cycle alias found for variables {seen}")
            if not name in aliases:
                return Variable(name)
            match aliases[name]:
                case Variable(aliased):
                    result[name] = chase(aliased, seen | {name})
                case aliased:
                    result[name] = aliased
            return result[name]

        for name in aliases:
            chase(name)
        return result

    def optimise(self):
        blocks = self.function.body.named_blocks()
        known_copies: dict[StellaIdent, Operand] = {}

        for b in blocks:
            for phi in blocks[b].phi_nodes:
                if (c := self.is_phi_trivial(phi)) is not None:
                    known_copies[phi.result.name] = c
            for instr in blocks[b].instructions:
                if (c := self.is_instruction_alias(instr)) is not None:
                    known_copies[instr.result.name] = c

        unprocessed_copies = len(known_copies) > 0
        while unprocessed_copies:
            unprocessed_copies = False
            effective_alias = self.chase_aliases(known_copies)

            def propagate_copy(operand: Operand):
                if not isinstance(operand, Variable):
                    return operand
                if operand.name in effective_alias:
                    return effective_alias[operand.name]
                else:
                    return operand

            for b in blocks:
                for i, phi in enumerate(blocks[b].phi_nodes):
                    blocks[b].phi_nodes[i] = phi.map_operands(propagate_copy)
                    phi = blocks[b].phi_nodes[i]
                    if (
                        not phi.result.name in known_copies
                        and (c := self.is_phi_trivial(phi)) is not None
                    ):
                        known_copies[phi.result.name] = c
                        unprocessed_copies = True
                for i, instr in enumerate(blocks[b].instructions):
                    blocks[b].instructions[i] = Instructions.map_args(instr, propagate_copy)
                    instr = blocks[b].instructions[i]
                    if (
                        isinstance(instr, UnOp)
                        and not instr.result.name in known_copies
                        and (c := self.is_instruction_alias(instr)) is not None
                    ):
                        known_copies[instr.result.name] = c
                        unprocessed_copies = True

                blocks[b].terminator = Terminators.map_operands(
                    blocks[b].terminator, propagate_copy
                )

        for b in blocks:
            blocks[b].phi_nodes = [
                p for p in blocks[b].phi_nodes if p.result.name not in known_copies
            ]
            blocks[b].instructions = [
                i
                for i in blocks[b].instructions
                if not isinstance(i, UnOp) or i.result.name not in known_copies
            ]

        return self.function
