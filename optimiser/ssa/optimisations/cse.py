from copy import deepcopy
from dataclasses import dataclass, field

from optimiser.ir.expressions import (
    BinOp,
    BinOpCode,
    Binding,
    BlockId,
    BoolConst,
    Callable,
    Instruction,
    Instructions,
    IntConst,
    Operand,
    Phi,
    StellaIdent,
    StellaType,
    Terminators,
    UnOp,
    UnOpCode,
    Variable,
)
from optimiser.ir.runtime import StellaBool, StellaInt
from optimiser.ssa.lengauer_tarjan import lengauer_tarjan


class ValueNumeration:
    @dataclass(frozen=True)
    class NumberedBinOp:
        op: BinOpCode
        result: StellaType
        lhs: int
        rhs: int

    @dataclass(frozen=True)
    class NumberedUnOp:
        op: UnOpCode
        result: StellaType
        arg: int

    @dataclass(frozen=True)
    class FreshVariable:
        name: StellaIdent

    type NumberedExpression = NumberedBinOp | NumberedUnOp
    type AbsrtractExpression = NumberedExpression | FreshVariable

    @dataclass(frozen=True)
    class Precalculated:
        number: int
        representative: Operand

    class ValuesTable:
        def __init__(self):
            self.tables: list[
                dict[ValueNumeration.AbsrtractExpression, ValueNumeration.Precalculated]
            ] = [{}]
            self.operands: dict[Operand, int] = {}
            self.representatives: dict[int, Operand] = {}
            self.fresh_counter = 0

        def enter(self):
            self.tables.append({})

        def leave(self):
            self.tables.pop()

        def fresh(self) -> int:
            self.fresh_counter += 1
            return self.fresh_counter - 1

        def bind_expr(
            self, expr: ValueNumeration.AbsrtractExpression, representative: Operand
        ) -> int:
            if self.find_expr(expr) is not None:
                raise KeyError(f"Double binding for expression {expr}")
            fresh = self.fresh()
            self.tables[-1][expr] = ValueNumeration.Precalculated(fresh, representative)
            self.representatives[fresh] = representative
            return fresh

        def bind_const(self, const: IntConst | BoolConst) -> int:
            if const in self.tables[0]:
                return self.tables[0][const].number
            fresh = self.fresh()
            self.tables[0][const] = ValueNumeration.Precalculated(fresh, const)
            self.representatives[fresh] = const
            return fresh

        def bind_operand(self, operand: Operand, value: int) -> int:
            if operand in self.operands:
                raise ValueError(f"Double binding for operand {operand}")
            self.operands[operand] = value
            return value

        def process_operand(self, operand) -> int:
            match operand:
                case Variable(name):
                    if not operand in self.operands:
                        raise (ValueError(f"Unexpected variable {name}"))
                    return self.operands[operand]
                case BoolConst() | IntConst():
                    if not operand in self.operands:
                        self.bind_operand(operand, self.bind_const(operand))
                    return self.operands[operand]
                case _:
                    raise ValueError(f"Unknown operand {operand}")

        def find_expr(
            self, expr: ValueNumeration.NumberedExpression
        ) -> ValueNumeration.Precalculated | None:
            for table in reversed(self.tables):
                if expr in table:
                    return table[expr]
            return None

        def process_expression(
            self, expr: ValueNumeration.NumberedExpression, representative: Operand
        ) -> int:
            if (found := self.find_expr(expr)) is not None:
                return found.number
            else:
                return self.bind_expr(expr, representative)

        def process_fresh_assignment(self, result: Binding):
            index = self.bind_fresh_variable(result.name)
            self.bind_operand(Variable(result.name), index)

        def process_phi(self, phi: Phi):
            self.process_fresh_assignment(phi.result)

        def process_assignment(self, instr: UnOp | BinOp):
            match instr:
                case UnOp(op="copy", arg=arg, result=result):
                    index = self.process_operand(arg)
                    self.bind_operand(Variable(name=result.name), index)
                case UnOp(op=op, arg=arg, result=result):
                    arg_index = self.process_operand(arg)
                    expr = ValueNumeration.NumberedUnOp(op=op, result=result.type, arg=arg_index)
                    expr_index = self.process_expression(expr, Variable(result.name))
                    self.bind_operand(Variable(name=result.name), expr_index)
                case BinOp(op=op, lhs=lhs, rhs=rhs, result=result):
                    lhs_index = self.process_operand(lhs)
                    rhs_index = self.process_operand(rhs)
                    expr = ValueNumeration.NumberedBinOp(
                        op=op, lhs=lhs_index, rhs=rhs_index, result=result.type
                    )
                    expr_index = self.process_expression(expr, Variable(name=result.name))
                    self.bind_operand(Variable(name=result.name), expr_index)
                case _:
                    raise (ValueError(f"{instr} is not assignment"))

        def bind_fresh_variable(self, name: StellaIdent) -> int:
            return self.bind_expr(ValueNumeration.FreshVariable(name), Variable(name))


class CSE:
    NAME = "CSE"

    def __init__(self, function: Callable):
        if function.body.form != "ssa":
            raise ValueError(f"CSE with called on (non-SSA) {function.body.form} form")
        self.function = deepcopy(function)
        self.vars = set(self.function.params) | self.function.body.all_bindings()

    def optimise(self) -> Callable:
        blocks = self.function.body.named_blocks()
        dominators_tree = lengauer_tarjan(self.function.body)
        children = dominators_tree.children()
        numeration_table = ValueNumeration.ValuesTable()

        for param in self.function.params:
            numeration_table.process_fresh_assignment(param)

        def dfs(b: BlockId):
            numeration_table.enter()
            for phi in blocks[b].phi_nodes:
                numeration_table.process_phi(phi)

            for instruction in blocks[b].instructions:
                match instruction:
                    case UnOp() | BinOp():
                        numeration_table.process_assignment(instruction)
                    case _:
                        for result in Instructions.defined_vars(instruction):
                            numeration_table.process_fresh_assignment(result)

            for child in children[b]:
                dfs(child)

            numeration_table.leave()

        dfs(self.function.body.entry)

        def substitute_variables(operand: Operand):
            match operand:
                case Variable():
                    return numeration_table.representatives[
                        numeration_table.process_operand(operand)
                    ]
                case _:
                    return operand

        def is_unique(instr: Instruction) -> bool:
            match instr:
                case UnOp() | BinOp():
                    return numeration_table.representatives[
                        numeration_table.process_operand(Variable(instr.result.name))
                    ] == Variable(instr.result.name)
                case _:
                    return True

        for b in blocks:
            blocks[b].phi_nodes = [
                phi.map_operands(substitute_variables) for phi in blocks[b].phi_nodes
            ]
            blocks[b].instructions = [
                Instructions.map_args(instr, substitute_variables)
                for instr in blocks[b].instructions
                if is_unique(instr)
            ]
            blocks[b].terminator = Terminators.map_operands(
                blocks[b].terminator, substitute_variables
            )

        return self.function
