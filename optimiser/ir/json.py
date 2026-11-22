from __future__ import annotations

import json
from typing import Any, ClassVar

from .runtime import StellaInt, StellaBool
from .expressions import (
    Binding,
    Variable,
    IntConst,
    BoolConst,
    Operand,
    UnOp,
    Form,
    BinOp,
    Call,
    Run,
    StellaType,
    Runtime,
    Instruction,
    Phi,
    Jump,
    Branch,
    Return,
    Tail,
    TailEntry,
    Resume,
    Fail,
    Terminator,
    Block,
    ControlFlow,
    Callable,
    Layout,
    Frame,
    Unit,
)


class JsonParser:
    UN_OPS: ClassVar[frozenset[str]] = frozenset({"copy", "not"})
    BIN_OPS: ClassVar[frozenset[str]] = frozenset(
        {
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
        }
    )

    @classmethod
    def from_string(cls, text: str) -> Unit:
        return cls().parse_unit(json.loads(text))

    @classmethod
    def from_dict(cls, j: dict[str, Any]) -> Unit:
        return cls().parse_unit(j)

    def parse_unit(self, j: dict[str, Any]) -> Unit:
        form: Form = j["form"]
        if form not in ("cfg", "ssa"):
            raise NotImplementedError(f"form {form!r} is not supported")
        return Unit(
            format=j["format"],
            version=j["version"],
            profile=j["profile"],
            form=form,
            exports=list(j["exports"]),
            layouts=[self.parse_layout(x) for x in j["layouts"]],
            frames=[self.parse_frame(x) for x in j["frames"]],
            callables=[self.parse_callable(x, form) for x in j["callables"]],
        )

    def parse_layout(self, j: dict[str, Any]) -> Layout:
        return Layout(
            name=j["name"],
            fields=[self.parse_type(t) for t in j["fields"]],
            capacity=int(j["capacity"]),
            entry=j["entry"],
            updatable=j["updatable"],
        )

    def parse_frame(self, j: dict[str, Any]) -> Frame:
        return Frame(
            name=j["name"],
            saved=[self.parse_type(t) for t in j["saved"]],
            result=self.parse_type(j["result"]),
            target=j["target"],
        )

    def parse_callable(self, j: dict[str, Any], form: Form) -> Callable:
        kind = j["kind"]
        body = ControlFlow(
            form=form,
            locals=[self.parse_binding(b) for b in j.get("locals", [])],
            entry=j["entry"],
            blocks=[self.parse_block(b) for b in j["blocks"]],
        )
        return Callable(
            kind=kind,
            name=j["name"],
            params=[self.parse_binding(b) for b in j["params"]],
            result=self.parse_type(j["result"]) if kind == "function" else None,
            body=body,
        )

    def parse_block(self, j: dict[str, Any]) -> Block:
        phis: list[Phi] = []
        instrs: list[Instruction] = []
        seen_non_phi = False
        for item in j["instructions"]:
            if item["op"] == "phi":
                if seen_non_phi:
                    raise ValueError(f"phi after non-phi instruction in block {j['name']!r}")
                phis.append(self.parse_phi(item))
            else:
                seen_non_phi = True
                instrs.append(self.parse_instruction(item))
        return Block(
            name=j["name"],
            phi_nodes=phis,
            instructions=instrs,
            terminator=self.parse_terminator(j["terminator"]),
        )

    def parse_terminator(self, j: dict[str, Any]) -> Terminator:
        match j["op"]:
            case "jump":
                return Jump("jump", j["target"])
            case "branch":
                return Branch(
                    "branch",
                    self.parse_operand(j["args"][0]),
                    j["then"],
                    j["else"],
                )
            case "return":
                return Return("return", self.parse_operand(j["args"][0]))
            case "tail":
                return Tail(
                    "tail",
                    j["target"],
                    [self.parse_operand(a) for a in j["args"]],
                )
            case "tail-entry":
                return TailEntry(
                    "tail-entry",
                    [self.parse_type(t) for t in j["signature"]],
                    [self.parse_operand(a) for a in j["args"]],
                )
            case "resume":
                return Resume("resume", self.parse_operand(j["args"][0]))
            case "fail":
                return Fail("fail", j["error"])
        raise ValueError(f"unknown terminator op: {j['op']!r}")

    def parse_instruction(self, j: dict[str, Any]) -> Instruction:
        op = j["op"]
        if op in self.UN_OPS:
            return UnOp(
                op,
                self.parse_operand(j["args"][0]),
                self.parse_binding(j["results"][0]),
            )
        if op in self.BIN_OPS:
            return BinOp(
                op,
                self.parse_operand(j["args"][0]),
                self.parse_operand(j["args"][1]),
                self.parse_binding(j["results"][0]),
            )
        match op:
            case "call":
                return Call(
                    "call",
                    j["callee"],
                    [self.parse_operand(a) for a in j["args"]],
                    self.parse_binding(j["results"][0]),
                )
            case "run":
                return Run(
                    "run",
                    j["target"],
                    [self.parse_operand(a) for a in j["args"]],
                    self.parse_binding(j["results"][0]),
                )
            case "runtime":
                return Runtime(
                    "runtime",
                    j["name"],
                    [self.parse_operand(a) for a in j["args"]],
                    [self.parse_binding(b) for b in j["results"]],
                    j["attributes"],
                )
        raise ValueError(f"unknown instruction op: {op!r}")

    def parse_phi(self, j: dict[str, Any]) -> Phi:
        incoming: dict[str, Operand] = {}
        for inc in j["incoming"]:
            pred = inc["pred"]
            if pred in incoming:
                raise ValueError(f"duplicate phi predecessor: {pred!r}")
            incoming[pred] = self.parse_operand(inc["value"])
        return Phi("phi", self.parse_binding(j["results"][0]), incoming)

    def parse_operand(self, j: Any) -> Operand:
        match j:
            case {"var": name}:
                return Variable(name)
            case {"i64": s}:
                return IntConst(StellaInt.from_str(s))
            case {"bool": b}:
                return BoolConst(StellaBool(b))
        raise ValueError(f"invalid operand: {j!r}")

    def parse_binding(self, j: dict[str, Any]) -> Binding:
        return Binding(j["name"], self.parse_type(j["type"]))

    def parse_type(self, j: str) -> StellaType:
        return j


class JsonEmitter:
    @classmethod
    def to_dict(cls, unit: Unit) -> dict[str, Any]:
        return cls().emit_unit(unit)

    @classmethod
    def to_string(cls, unit: Unit, *, indent: int | None = 2) -> str:
        return json.dumps(
            cls().emit_unit(unit),
            ensure_ascii=False,
            indent=indent,
            separators=None if indent is not None else (",", ":"),
        )

    def emit_unit(self, u: Unit) -> dict[str, Any]:
        return {
            "format": u.format,
            "version": u.version,
            "profile": u.profile,
            "form": u.form,
            "exports": list(u.exports),
            "layouts": [self.emit_layout(x) for x in u.layouts],
            "frames": [self.emit_frame(x) for x in u.frames],
            "callables": [self.emit_callable(x) for x in u.callables],
        }

    def emit_layout(self, l: Layout) -> dict[str, Any]:
        return {
            "name": l.name,
            "fields": list(l.fields),
            "capacity": str(l.capacity),
            "entry": l.entry,
            "updatable": l.updatable,
        }

    def emit_frame(self, f: Frame) -> dict[str, Any]:
        return {
            "name": f.name,
            "saved": list(f.saved),
            "result": f.result,
            "target": f.target,
        }

    def emit_callable(self, c: Callable) -> dict[str, Any]:
        d: dict[str, Any] = {
            "kind": "function" if c.result is not None else "routine",
            "name": c.name,
            "params": [self.emit_binding(b) for b in c.params],
        }
        if c.result is not None:
            d["result"] = c.result
        d.update(self.emit_control_flow(c.body))
        return d

    def emit_control_flow(self, b: ControlFlow) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if b.form == "cfg":
            d["locals"] = [self.emit_binding(x) for x in b.locals]
        d["entry"] = b.entry
        d["blocks"] = [self.emit_block(x) for x in b.blocks]
        return d

    def emit_block(self, b: Block) -> dict[str, Any]:
        return {
            "name": b.name,
            "instructions": [
                *[self.emit_phi(p) for p in b.phi_nodes],
                *[self.emit_instruction(i) for i in b.instructions],
            ],
            "terminator": self.emit_terminator(b.terminator),
        }

    def emit_terminator(self, t: Terminator) -> dict[str, Any]:
        match t:
            case Jump(target=target):
                return {"op": "jump", "target": target}
            case Branch(condition=cond, on_true=on_true, on_false=on_false):
                return {
                    "op": "branch",
                    "args": [self.emit_operand(cond)],
                    "then": on_true,
                    "else": on_false,
                }
            case Return(value=v):
                return {"op": "return", "args": [self.emit_operand(v)]}
            case Tail(target=target, args=args):
                return {
                    "op": "tail",
                    "target": target,
                    "args": [self.emit_operand(a) for a in args],
                }
            case TailEntry(signature=sig, args=args):
                return {
                    "op": "tail-entry",
                    "signature": list(sig),
                    "args": [self.emit_operand(a) for a in args],
                }
            case Resume(value=v):
                return {"op": "resume", "args": [self.emit_operand(v)]}
            case Fail(error=err):
                return {"op": "fail", "error": err}
        raise TypeError(f"unknown terminator: {type(t).__name__}")

    def emit_instruction(self, i: Instruction) -> dict[str, Any]:
        match i:
            case UnOp(op=op, arg=arg, result=r):
                return {
                    "op": op,
                    "args": [self.emit_operand(arg)],
                    "results": [self.emit_binding(r)],
                }
            case BinOp(op=op, lhs=lhs, rhs=rhs, result=r):
                return {
                    "op": op,
                    "args": [self.emit_operand(lhs), self.emit_operand(rhs)],
                    "results": [self.emit_binding(r)],
                }
            case Call(callee=callee, args=args, result=r):
                return {
                    "op": "call",
                    "callee": callee,
                    "args": [self.emit_operand(a) for a in args],
                    "results": [self.emit_binding(r)],
                }
            case Run(target=target, args=args, result=r):
                return {
                    "op": "run",
                    "target": target,
                    "args": [self.emit_operand(a) for a in args],
                    "results": [self.emit_binding(r)],
                }
            case Runtime(name=name, args=args, results=rs, attributes=attrs):
                return {
                    "op": "runtime",
                    "name": name,
                    "args": [self.emit_operand(a) for a in args],
                    "results": [self.emit_binding(b) for b in rs],
                    "attributes": attrs,
                }
        raise TypeError(f"unknown instruction: {type(i).__name__}")

    def emit_phi(self, p: Phi) -> dict[str, Any]:
        return {
            "op": "phi",
            "results": [self.emit_binding(p.result)],
            "incoming": [
                {"pred": pred, "value": self.emit_operand(v)} for pred, v in p.incoming.items()
            ],
        }

    def emit_binding(self, b: Binding) -> dict[str, Any]:
        return {"name": b.name, "type": b.type}

    def emit_operand(self, o: Operand) -> dict[str, Any]:
        match o:
            case Variable(name=name):
                return {"var": name}
            case IntConst(value=v):
                return {"i64": v.to_str()}
            case BoolConst(value=b):
                return {"bool": b.v}
        raise TypeError(f"unknown operand: {type(o).__name__}")
