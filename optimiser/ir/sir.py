from __future__ import annotations

import json
from typing import Any, assert_never

from optimiser.ir.expressions import (
    Binding,
    Variable,
    IntConst,
    BoolConst,
    Operand,
    UnOp,
    BinOp,
    Call,
    Run,
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
from optimiser.ir.runtime import StellaBool


def json_quote(s: str) -> str:
    return json.dumps(s, ensure_ascii=False)


class SirEmitter:
    @classmethod
    def to_string(cls, unit: Unit) -> str:
        return cls().emit_unit(unit)

    def emit_unit(self, u: Unit) -> str:
        if u.form not in ("cfg", "ssa"):
            raise NotImplementedError(f"form {u.form!r} is not supported")
        parts: list[str] = [f"stella-ir {u.version} {u.form} {json_quote(u.profile)};"]
        if u.exports:
            parts.append(self.emit_exports(u.exports))
        parts.extend(self.emit_layout(l) for l in u.layouts)
        parts.extend(self.emit_frame(f) for f in u.frames)
        parts.extend(self.emit_callable(c) for c in u.callables)
        return "\n".join(parts) + "\n"

    def emit_exports(self, names: list[str]) -> str:
        return "export " + ", ".join(f"@{n}" for n in names) + ";"

    def emit_layout(self, l: Layout) -> str:
        entry = f"@{l.entry}" if l.entry is not None else "none"
        upd = "true" if l.updatable else "false"
        return (
            f"layout ${l.name} fields {self.types(l.fields)} "
            f"capacity {l.capacity} updatable {upd} entry {entry};"
        )

    def emit_frame(self, f: Frame) -> str:
        return (
            f"frame !{f.name} saved {self.types(f.saved)} " f"result {f.result} target @{f.target};"
        )

    def emit_callable(self, c: Callable) -> str:
        params = self.params(c.params)
        if c.result is not None:
            header = f"function @{c.name} {params} -> {c.result}"
        else:
            header = f"routine @{c.name} {params}"
        body = self.emit_control_flow(c.body)
        return header + " {\n" + self.indent(body, 4) + "\n}"

    def emit_control_flow(self, b: ControlFlow) -> list[str]:
        lines: list[str] = []
        if b.form == "cfg":
            lines.append(f"locals {self.params(b.locals)};")
        lines.append(f"entry ^{b.entry};")
        for blk in b.blocks:
            lines.extend(self.emit_block(blk))
        return lines

    def emit_block(self, b: Block) -> list[str]:
        lines: list[str] = [f"^{b.name}:"]
        lines.extend("  " + self.emit_phi(p) for p in b.phi_nodes)
        lines.extend("  " + self.emit_instruction(i) for i in b.instructions)
        lines.append("  " + self.emit_terminator(b.terminator))
        return lines

    def emit_phi(self, p: Phi) -> str:
        incoming = ", ".join(f"^{pred}: {self.atom(v)}" for pred, v in p.incoming.items())
        return f"{self.binding(p.result)} = phi {{ {incoming} }};"

    def emit_instruction(self, i: Instruction) -> str:
        match i:
            case UnOp(op=op, arg=arg, result=r):
                return f"{self.results([r])} = {op}{self.args([arg])};"
            case BinOp(op=op, lhs=lhs, rhs=rhs, result=r):
                return f"{self.results([r])} = {op}{self.args([lhs, rhs])};"
            case Call(callee=callee, args=args, result=r):
                return f"{self.results([r])} = call @{callee}{self.args(args)};"
            case Run(target=target, args=args, result=r):
                return f"{self.results([r])} = run @{target}{self.args(args)};"
            case Runtime(name=name, args=args, results=rs, attributes=attrs):
                return (
                    f"{self.results(rs)} = runtime {json_quote(name)}"
                    f"{self.args(args)} {self.attributes(attrs)};"
                )
            case _:
                assert False

    def emit_terminator(self, t: Terminator) -> str:
        match t:
            case Jump(target=target):
                return f"jump ^{target};"
            case Branch(condition=c, on_true=a, on_false=b):
                return f"branch {self.atom(c)}, ^{a}, ^{b};"
            case Return(value=v):
                return f"return {self.atom(v)};"
            case Tail(target=target, args=args):
                return f"tail @{target}{self.args(args)};"
            case TailEntry(signature=sig, args=args):
                return f"tail-entry {self.types(sig)} {self.args(args)};"
            case Resume(value=v):
                return f"resume {self.atom(v)};"
            case Fail(error=err):
                return f"fail {json_quote(err)};"
            case _:
                assert False

    def binding(self, b: Binding) -> str:
        return f"%{b.name}: {b.type}"

    def params(self, params: list[Binding]) -> str:
        return "(" + ", ".join(self.binding(p) for p in params) + ")"

    def results(self, results: list[Binding]) -> str:
        return "[" + ", ".join(self.binding(r) for r in results) + "]"

    def types(self, types: list[str]) -> str:
        return "(" + ", ".join(types) + ")"

    def args(self, args: list[Operand]) -> str:
        return "(" + ", ".join(self.atom(a) for a in args) + ")"

    def atom(self, a: Operand) -> str:
        match a:
            case Variable(name=name):
                return f"%{name}"
            case IntConst(value=v):
                return v.to_str()
            case BoolConst(value):
                return value.to_str()
            case _:
                assert False

    def attributes(self, attrs: dict[str, Any]) -> str:
        if not attrs:
            return "{}"
        items = ", ".join(f"{json_quote(k)}: {self.static(v)}" for k, v in attrs.items())
        return "{ " + items + " }"

    def static(self, v: Any) -> str:
        if isinstance(v, str):
            return json_quote(v)
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, list):
            return "[" + ", ".join(self.static(x) for x in v) + "]"
        if isinstance(v, dict):
            return self.attributes(v)
        raise TypeError(f"invalid static value: {v!r}")

    @staticmethod
    def indent(lines: list[str], n: int = 1) -> str:
        pad = "  " * n
        return "\n".join(pad + line for line in lines)
