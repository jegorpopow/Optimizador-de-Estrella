from enum import Enum
from typing import Literal, final
from copy import deepcopy

from optimiser.ir.expressions import (
    BinOp,
    Block,
    BlockId,
    BoolConst,
    Branch,
    Call,
    Callable,
    ControlFlow,
    Edge,
    Instruction,
    Instructions,
    IntConst,
    Jump,
    Location,
    Operand,
    Phi,
    Run,
    Runtime,
    StellaIdent,
    Terminator,
    Terminators,
    UnOp,
    Usage,
    Variable,
)
from optimiser.ir.runtime import StellaBool, StellaInt, StellaUnop, StellaBinop


class AbstractEvaluation:
    @final
    class BottomType(Enum):
        BOTTOM = "⊥"

        def __repr__(self) -> str:
            return "⊥"

    @final
    class TopType(Enum):
        TOP = "⊤"

        def __repr__(self) -> str:
            return "⊤"

    BOTTOM: Literal[BottomType.BOTTOM] = BottomType.BOTTOM
    TOP: Literal[TopType.TOP] = TopType.TOP
    type Value = StellaInt | StellaBool | TopType | BottomType
    type State = dict[StellaIdent, Value]

    @staticmethod
    def is_concrete(v):
        return v != AbstractEvaluation.TOP and v != AbstractEvaluation.BOTTOM

    @staticmethod
    def evaluate_operand(operand: Operand, state: State) -> Value:
        if not isinstance(operand, Variable):
            return operand.value
        return state[operand.name]

    @staticmethod
    def merge_values(values: set[Value]) -> Value:
        values = values - {AbstractEvaluation.BOTTOM}
        if not values:
            return AbstractEvaluation.BOTTOM
        if AbstractEvaluation.TOP in values or len(values) > 1:
            return AbstractEvaluation.TOP
        return next(iter(values))

    @staticmethod
    def evaluate(instruction: Instruction, state: State) -> list[Value]:
        match instruction:
            case UnOp(op, arg, _):
                argument = AbstractEvaluation.evaluate_operand(arg, state)
                if op == "copy":
                    return [argument]
                else:
                    if AbstractEvaluation.is_concrete(argument):
                        return [StellaUnop.perform(op, argument)]
                    else:
                        return [argument]
            case BinOp(op, lhs, rhs, _):
                lhs_value = AbstractEvaluation.evaluate_operand(lhs, state)
                rhs_value = AbstractEvaluation.evaluate_operand(rhs, state)
                if AbstractEvaluation.is_concrete(lhs_value) and AbstractEvaluation.is_concrete(
                    rhs_value
                ):
                    try:
                        return [StellaBinop.perform(op, lhs_value, rhs_value)]
                    except ArithmeticError:
                        return [AbstractEvaluation.TOP]
                elif lhs_value is AbstractEvaluation.TOP or rhs_value is AbstractEvaluation.TOP:
                    return [AbstractEvaluation.TOP]
                else:
                    return [AbstractEvaluation.BOTTOM]
            case Call():
                return [AbstractEvaluation.TOP]
            case Run():
                return [AbstractEvaluation.TOP]
            case Runtime(results=results):
                return [AbstractEvaluation.TOP for _ in results]
            case _:
                assert False


class SCCP:
    NAME = "SCCP"

    def __init__(self, function: Callable):
        if function.body.form != "ssa":
            raise ValueError(f"SCCP with called on (non-SSA) {function.body.form} form")
        self.function = function
        self.vars = set(self.function.params) | self.function.body.all_bindings()

    def analyse(self):
        blocks = self.function.body.named_blocks()
        edges_to_process: list[Edge] = []
        vars_to_process: list[StellaIdent] = []
        params_set = set(self.function.params)
        usages = self.function.body.all_usages({var.name for var in self.vars})

        model: AbstractEvaluation.State = {
            var.name: AbstractEvaluation.BOTTOM if not var in params_set else AbstractEvaluation.TOP
            for var in self.vars
        }
        enters: dict[BlockId, set[BlockId]] = {b: set() for b in blocks}

        def promote(var: StellaIdent, val: AbstractEvaluation.Value):
            if val != model[var]:
                model[var] = val
                vars_to_process.append(var)

        def recalculate_phi(b: BlockId, phi: Phi):
            reachable = {
                AbstractEvaluation.evaluate_operand(phi.incoming[p], model)
                for p in phi.incoming
                if p in enters[b]
            }
            promote(phi.result.name, AbstractEvaluation.merge_values(reachable))

        def recalculate_instruction(_: BlockId, instruction: Instruction):
            values = AbstractEvaluation.evaluate(instruction, model)
            for result, value in zip(Instructions.defined_vars(instruction), values):
                promote(result.name, value)

        def recalculate_terminator(b: BlockId, term: Terminator):
            edges = set()
            match term:
                case Jump(target=target):
                    edges = {target}
                case Branch(condition=condition, on_true=t, on_false=f):
                    condition_value = AbstractEvaluation.evaluate_operand(condition, model)
                    if AbstractEvaluation.is_concrete(condition_value):
                        assert isinstance(
                            condition_value, StellaBool
                        ), f"{condition_value} is not StellaBool"
                        if condition_value.v:
                            edges = {t}
                        else:
                            edges = {f}
                    elif condition_value is AbstractEvaluation.TOP:
                        edges = {t, f}
                    elif condition_value is AbstractEvaluation.BOTTOM:
                        edges = set()
                case _:
                    edges = set()
            for dest in edges:
                edges_to_process.append(Edge(b, dest))

        def recalculate_usage(usage: Usage):
            obj = blocks[usage.block].at(usage.place)
            match usage.place:
                case Location.Phi():
                    recalculate_phi(usage.block, obj)
                case Location.Instruction():
                    recalculate_instruction(usage.block, obj)
                case Location.Terminator():
                    recalculate_terminator(usage.block, obj)
                case _:
                    raise ValueError(f"Unknown location {usage.place}")

        edges_to_process = [Edge(None, self.function.body.entry)]
        while len(edges_to_process) != 0 or len(vars_to_process) != 0:
            if len(edges_to_process) != 0:
                edge = edges_to_process.pop()
                if not edge.source in enters[edge.destination]:
                    first_enter = len(enters[edge.destination]) == 0
                    enters[edge.destination].add(edge.source)
                    for phi in blocks[edge.destination].phi_nodes:
                        recalculate_phi(edge.destination, phi)
                    if first_enter:
                        for instruction in blocks[edge.destination].instructions:
                            recalculate_instruction(edge.destination, instruction)
                        recalculate_terminator(
                            edge.destination, blocks[edge.destination].terminator
                        )
            else:
                var = vars_to_process.pop()
                for usage in usages[var]:
                    recalculate_usage(usage)

        enters[self.function.body.entry].remove(None)
        return model, enters

    def optimise(self) -> Callable:
        model, enters = self.analyse()

        # print(f"model = {model}")

        def simplify_terminator(terminator: Terminator) -> Terminator:
            if not isinstance(terminator, Branch):
                return deepcopy(terminator)
            else:
                condition_value = AbstractEvaluation.evaluate_operand(terminator.condition, model)
                if not AbstractEvaluation.is_concrete(condition_value):
                    return deepcopy(terminator)
                assert isinstance(condition_value, StellaBool)
                if condition_value.v:
                    return Jump(op="jump", target=terminator.on_true)
                else:
                    return Jump(op="jump", target=terminator.on_false)

        def simplify_operand(operand: Operand) -> Operand:
            if not isinstance(operand, Variable):
                return deepcopy(operand)
            if AbstractEvaluation.is_concrete(model[operand.name]):
                match model[operand.name]:
                    case StellaInt():
                        return IntConst(model[operand.name])
                    case StellaBool():
                        return BoolConst(model[operand.name])
                    case _:
                        raise ValueError(f"Unknown concrete value {model[operand.name]}")
            else:
                return deepcopy(operand)

        blocks = [
            Block(
                name=old.name,
                phi_nodes=[
                    Phi(
                        op="phi",
                        result=phi.result,
                        incoming={
                            p: simplify_operand(phi.incoming[p])
                            for p in phi.incoming
                            if p in enters[old.name]
                        },
                    )
                    for phi in old.phi_nodes
                    if not AbstractEvaluation.is_concrete(model[phi.result.name])
                ],
                instructions=[
                    Instructions.map_args(i, simplify_operand)
                    for i in old.instructions
                    if any(
                        not AbstractEvaluation.is_concrete(model[v.name])
                        for v in Instructions.defined_vars(i)
                    )
                ],
                terminator=Terminators.map_operands(
                    simplify_terminator(old.terminator), simplify_operand
                ),
            )
            for old in self.function.body.blocks
            if old.name == self.function.body.entry or len(enters[old.name]) > 0
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
                blocks=blocks,
            ),
        )
