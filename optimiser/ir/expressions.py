from dataclasses import dataclass, field
from typing import Literal
from .runtime import StellaBool, StellaInt
from enum import StrEnum

StellaIdent = str
StellaError = str
BlockId = str


class StellaType(StrEnum):
    INT = "i64"
    BOOL = "bool"
    REF = "ref"

    def is_primitive(self) -> bool:
        return self == StellaType.INT or self == StellaType.BOOL


@dataclass(frozen=True)
class Binding:
    name: StellaIdent
    type: StellaType

    def rename(self, subst: Callable[[StellaIdent], StellaIdent]) -> Binding:
        return Binding(subst(self.name), self.type)


@dataclass(frozen=True)
class Variable:
    name: StellaIdent


@dataclass(frozen=True)
class IntConst:
    value: StellaInt


@dataclass(frozen=True)
class BoolConst:
    value: StellaBool


type Operand = Variable | IntConst | BoolConst
type Static = str | bool | list[Static] | dict[str, Static]


class Operands:
    @staticmethod
    def rename_vars(op: Operand, subst: Callable[[StellaIdent], StellaIdent]) -> Operand:
        match op:
            case Variable(name=name):
                return Variable(subst(name))
            case _:
                return op

    @staticmethod
    def vars(op: Operand) -> list[StellaIdent]:
        match op:
            case Variable(name=name):
                return [name]
            case _:
                return []


BinOpCode = Literal[
    "add",
    "sub",
    "mul",
    "div",
    "rem",
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
    "and",
    "or",
    "xor",
]

UnOpCode = Literal["not", "copy"]


@dataclass
class UnOp:
    op: UnOpCode
    arg: Operand
    result: Binding


@dataclass
class BinOp:
    op: BinOpCode
    lhs: Operand
    rhs: Operand
    result: Binding


@dataclass
class Call:
    op: Literal["call"]
    callee: StellaIdent
    args: list[Operand]
    result: Binding


@dataclass
class Run:
    op: Literal["run"]
    target: StellaIdent
    args: list[Operand]
    result: Binding


@dataclass
class Runtime:
    op: Literal["runtime"]
    name: str
    args: list[Operand]
    results: list[Binding]
    attributes: Static


type Instruction = UnOp | BinOp | Call | Run | Runtime


class Instructions:
    @staticmethod
    def map_args(instruction: Instruction, subst: Callable[[Operand], Operand]) -> Instruction:
        match instruction:
            case UnOp(op, arg, result):
                return UnOp(op, subst(arg), result)
            case BinOp(op, lhs, rhs, result):
                return BinOp(op, subst(lhs), subst(rhs), result)
            case Call(op, callee, args, result):
                return Call(op, callee, [subst(arg) for arg in args], result)
            case Run(op, target, args, result):
                return Run(op, target, [subst(arg) for arg in args], result)
            case Runtime(op, name, args, results, attributes):
                return Runtime(
                    op,
                    name,
                    [subst(arg) for arg in args],
                    results,
                    attributes,
                )

    @staticmethod
    def used_vars(instruction: Instruction) -> list[StellaIdent]:
        match instruction:
            case UnOp(arg=arg):
                return Operands.vars(arg)
            case BinOp(lhs=lhs, rhs=rhs):
                return Operands.vars(lhs) + Operands.vars(rhs)
            case Call(args=args):
                return [var for arg in args for var in Operands.vars(arg)]
            case Run(args=args):
                return [var for arg in args for var in Operands.vars(arg)]
            case Runtime(args=args):
                return [var for arg in args for var in Operands.vars(arg)]

    @staticmethod
    def rename_results(
        instruction: Instruction, subst: Callable[[StellaIdent], StellaIdent]
    ) -> Instruction:
        match instruction:
            case UnOp(op, arg, result):
                return UnOp(op, arg, result.rename(subst))
            case BinOp(op, lhs, rhs, result):
                return BinOp(op, lhs, rhs, result.rename(subst))
            case Call(op, callee, args, result):
                return Call(op, callee, args, result.rename(subst))
            case Run(op, target, args, result):
                return Run(op, target, args, result.rename(subst))
            case Runtime(op, name, args, results, attributes):
                return Runtime(
                    op,
                    name,
                    args,
                    [result.rename(subst) for result in results],
                    attributes,
                )

    @staticmethod
    def is_pure(instruction: Instruction) -> bool:
        match instruction:
            case UnOp():
                return True
            case BinOp(op=op, rhs=rhs):
                if op not in ["div", "rem"]:
                    return True
                else:
                    match rhs:
                        case IntConst(value):
                            return value.v != 0
                        case _:
                            return False
            case _:
                return False

    @staticmethod
    def defined_vars(instruction: Instruction) -> list[Binding]:
        match instruction:
            case Runtime(results=results):
                return results
            case _:
                return [instruction.result]

    @staticmethod
    def is_instruction(obj) -> bool:
        return isinstance(obj, (UnOp, BinOp, Call, Run, Runtime))


@dataclass
class Phi:
    op: Literal["phi"]
    result: Binding
    incoming: dict[StellaIdent, Operand]

    def used_vars(self) -> list[StellaIdent]:
        return [var for p in self.incoming for var in Operands.vars(self.incoming[p])]

    def map_operands(self, subst: Callable[[Operand], Operand]) -> Phi:
        return Phi(
            op="phi",
            result=self.result,
            incoming={p: subst(self.incoming[p]) for p in self.incoming},
        )

    @staticmethod
    def is_phi(obj):
        return isinstance(obj, Phi)


type SSAInstruction = Phi

type GeneralisedInstruction = Instruction | SSAInstruction


@dataclass
class Jump:
    op: Literal["jump"]
    target: BlockId

    def successors(self) -> list[BlockId]:
        return [self.target]


@dataclass
class Branch:
    op: Literal["branch"]
    condition: Operand
    on_true: BlockId
    on_false: BlockId

    def successors(self) -> list[BlockId]:
        return [self.on_true, self.on_false]


@dataclass
class Return:
    op: Literal["return"]
    value: Operand

    def successors(self) -> list[BlockId]:
        return []


@dataclass
class Tail:
    op: Literal["tail"]
    target: StellaIdent
    args: list[Operand]

    def successors(self) -> list[BlockId]:
        return []


@dataclass
class TailEntry:
    op: Literal["tail-entry"]
    signature: list[StellaType]
    args: list[Operand]

    def successors(self) -> list[BlockId]:
        return []


@dataclass
class Resume:
    op: Literal["resume"]
    value: Operand

    def successors(self) -> list[BlockId]:
        return []


@dataclass
class Fail:
    op: Literal["fail"]
    error: StellaError

    def successors(self) -> list[BlockId]:
        return []


type Exit = Return | Tail | TailEntry | Resume | Fail
type Terminator = Jump | Branch | Exit


class Terminators:
    @staticmethod
    def map_operands(term: Terminator, subst: Callable[[Operand], Operand]) -> Terminator:
        match term:
            case Jump(_, _):
                return term
            case Branch(op, condition, on_true, on_false):
                return Branch(op, subst(condition), on_true, on_false)
            case Return(op, value):
                return Return(op, subst(value))
            case Tail(op, target, args):
                return Tail(op, target, [subst(arg) for arg in args])
            case TailEntry(op, signature, args):
                return TailEntry(op, signature, [subst(arg) for arg in args])
            case Resume(op, value):
                return Resume(op, subst(value))
            case Fail(_, _):
                return term

    @staticmethod
    def used_vars(term: Terminator) -> list[StellaIdent]:
        match term:
            case Jump(_, _):
                return []
            case Branch(condition=condition):
                return Operands.vars(condition)
            case Return(value=value):
                return Operands.vars(value)
            case Tail(args=args):
                return [var for arg in args for var in Operands.vars(arg)]
            case TailEntry(args=args):
                return [var for arg in args for var in Operands.vars(arg)]
            case Resume(value=value):
                return Operands.vars(value)
            case Fail(_, _):
                return []

    @staticmethod
    def redirect(term: Terminator, mapping: Callable[[BlockId], BlockId]) -> Terminator:
        match term:
            case Jump(op=op, target=target):
                return Jump(op, mapping(target))
            case Branch(op=op, condition=cond, on_true=on_true, on_false=on_flase):
                return Branch(
                    op=op, condition=cond, on_true=mapping(on_true), on_false=mapping(on_flase)
                )
            case _:
                return term

    @staticmethod
    def is_terminator(obj) -> bool:
        return isinstance(obj, (Jump, Branch, Return, Tail, TailEntry, Resume, Fail))


class Location:
    @dataclass(frozen=True)
    class Phi:
        index: int

    @dataclass(frozen=True)
    class Instruction:
        index: int

    @dataclass(frozen=True)
    class Terminator:
        pass

    type Loc = Phi | Instruction | Terminator


@dataclass
class Block:
    name: BlockId
    phi_nodes: list[Phi]
    instructions: list[Instruction]
    terminator: Terminator

    def defined_vars(self) -> list[Binding]:
        return [phi.result for phi in self.phi_nodes] + [
            var for op in self.instructions for var in Instructions.defined_vars(op)
        ]

    def at(self, location: Location.Loc) -> Phi | Instruction | Terminator:
        match location:
            case Location.Phi(i):
                return self.phi_nodes[i]
            case Location.Instruction(i):
                return self.instructions[i]
            case Location.Terminator():
                return self.terminator
            case _:
                raise ValueError(f"Unknown block inner-location: {location}")


@dataclass(frozen=True)
class Edge:
    source: BlockId
    destination: BlockId


Form = Literal["cfg", "ssa"]


@dataclass(frozen=True)
class Usage:
    block: BlockId
    place: Location.Loc


@dataclass
class ControlFlow:
    form: Form = "cfg"
    locals: list[Binding] = field(default_factory=list)
    entry: StellaIdent = ""
    blocks: list[Block] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.form == "ssa" and self.locals:
            raise ValueError("SSA body must not declare locals")

    def named_blocks(self) -> dict[BlockId, Block]:
        return {b.name: b for b in self.blocks}

    def dfs(
        self, on_enter=lambda *args: None, on_exit=lambda *args: None, on_edge=lambda *args: None
    ):
        grey = set()
        black = set()
        block_by_name = self.named_blocks()

        def visit(p: BlockId | None, b: BlockId):
            if b in grey or b in black:
                return

            on_enter(p, b)
            grey.add(b)

            children = block_by_name[b].terminator.successors()
            for child in children:
                on_edge(b, child)
                visit(b, child)

            on_exit(b)
            grey.remove(b)
            black.add(b)

        visit(None, self.entry)

    def all_bindings(self) -> set[Binding]:
        if self.form == "cfg":
            return set(self.locals)
        else:
            return set().union(*[set(b.defined_vars()) for b in self.blocks])

    def predecessors(self) -> dict[BlockId, set[BlockId]]:
        result = {b.name: set() for b in self.blocks}
        self.dfs(on_edge=lambda p, v: result[v].add(p))
        return result

    def successors(self) -> dict[BlockId, set[BlockId]]:
        result = {b.name: set() for b in self.blocks}
        self.dfs(on_edge=lambda p, v: result[p].add(v))
        return result

    def all_usages(self, vars: set[StellaIdent]) -> dict[StellaIdent, list[Usage]]:
        blocks = self.named_blocks()
        usages = {v: [] for v in vars}

        for b in blocks:
            for i, phi in enumerate(blocks[b].phi_nodes):
                for var in set(phi.used_vars()):
                    usages[var].append(Usage(b, Location.Phi(i)))
            for i, instr in enumerate(blocks[b].instructions):
                for var in set(Instructions.used_vars(instr)):
                    usages[var].append(Usage(b, Location.Instruction(i)))
            for var in set(Terminators.used_vars(blocks[b].terminator)):
                usages[var].append(Usage(b, Location.Terminator()))

        return usages

    def all_defs_ssa(self, vars: set[StellaIdent]) -> dict[StellaIdent, Usage | None]:
        if self.form != "ssa":
            raise ValueError(f"all_defs_ssa called on the graph of {self.form} form")
        blocks = self.named_blocks()
        defs = {v: None for v in vars}

        for b in blocks:
            for i, phi in enumerate(blocks[b].phi_nodes):
                defs[phi.result.name] = Usage(b, Location.Phi(i))
            for i, instr in enumerate(blocks[b].instructions):
                for var in Instructions.defined_vars(instr):
                    defs[var.name] = Usage(b, Location.Instruction(i))
        return defs

    def split_edge(self, edge: Edge, block_id: BlockId):
        self.blocks.append(
            Block(
                name=block_id,
                phi_nodes=[],
                instructions=[],
                terminator=Jump("jump", edge.destination),
            )
        )
        blocks = self.named_blocks()
        blocks[edge.source].terminator = Terminators.redirect(
            blocks[edge.source].terminator, lambda b: block_id if b == edge.destination else b
        )
        for phi in blocks[edge.destination].phi_nodes:
            operand = phi.incoming.pop(edge.source)
            phi.incoming[block_id] = operand


@dataclass
class Layout:
    name: StellaIdent
    fields: list[StellaType]
    capacity: int
    entry: StellaIdent | None
    updatable: bool


@dataclass
class Frame:
    name: StellaIdent
    saved: list[StellaType]
    result: StellaType
    target: StellaIdent


@dataclass
class Callable:
    kind: Literal["function", "routine"]
    name: StellaIdent
    params: list[Binding]
    result: StellaType | None
    body: ControlFlow


@dataclass
class Unit:
    format: Literal["stella-ir"]
    version: Literal["1"]
    profile: str
    form: Form
    exports: list[StellaIdent]
    layouts: list[Layout]
    frames: list[Frame]
    callables: list[Callable]
