class StellaInt:
    _MIN = -(1 << 63)
    _MAX = (1 << 63) - 1
    __slots__ = ("v",)
    __match_args__ = ("v",)

    def __init__(self, v: int) -> None:
        if not self._MIN <= v <= self._MAX:
            raise ValueError(f"i64 literal out of range: {v}")
        self.v = v

    @classmethod
    def from_str(cls, s: str) -> StellaInt:
        d = s[1:] if s.startswith("-") else s
        if s != "0" and not (d and d[0] in "123456789" and d.isascii() and d.isdigit()):
            raise ValueError(f"invalid i64 literal: `{s}`")
        return cls(int(s))

    def to_str(self) -> str:
        return str(self.v)

    def __eq__(self, o: object) -> bool:
        return isinstance(o, StellaInt) and self.v == o.v

    def __hash__(self) -> int:
        return hash(self.v)

    def __repr__(self) -> str:
        return f"StellaInt({self.v})"

    @classmethod
    def wrap(cls, v: int) -> StellaInt:
        v &= (1 << 64) - 1
        return cls(v - (1 << 64) if v >= (1 << 63) else v)

    @staticmethod
    def binary_operation(op, lhs: StellaInt, rhs: StellaInt) -> StellaInt | StellaBool:
        a, b = lhs.v, rhs.v
        match op:
            case "add":
                return StellaInt.wrap(a + b)
            case "sub":
                return StellaInt.wrap(a - b)
            case "mul":
                return StellaInt.wrap(a * b)
            case "div" | "rem":
                if b == 0:
                    raise ZeroDivisionError("arithmetic-zero")
                if a == StellaInt._MIN and b == -1:
                    q, r = StellaInt._MIN, 0
                else:
                    q = abs(a) // abs(b)
                    if (a < 0) != (b < 0):
                        q = -q
                    r = a - q * b
                return StellaInt(q if op == "div" else r)
            case "eq":
                return StellaBool(a == b)
            case "ne":
                return StellaBool(a != b)
            case "lt":
                return StellaBool(a < b)
            case "le":
                return StellaBool(a <= b)
            case "gt":
                return StellaBool(a > b)
            case "ge":
                return StellaBool(a >= b)
            case _:
                raise ValueError(f"`{op}` is not an i64 binary operation")

    @staticmethod
    def unary_operation(op, operand: StellaInt) -> StellaInt:
        match op:
            case "copy":
                return StellaInt(operand.v)
            case _:
                raise ValueError(f"{op!r} is not an i64 unary operation")


class StellaBool:
    __slots__ = ("v",)
    __match_args__ = ("v",)

    def __init__(self, v: bool) -> None:
        self.v = v

    def __eq__(self, o: object) -> bool:
        return isinstance(o, StellaBool) and self.v == o.v

    def __hash__(self) -> int:
        return hash(self.v)

    def __repr__(self) -> str:
        return f"StellaBool({self.v})"

    def to_str(self) -> str:
        return str(self.v)

    @staticmethod
    def binary_operation(op, lhs: StellaBool, rhs: StellaBool) -> StellaBool:
        match op:
            case "and":
                return StellaBool(lhs.v and rhs.v)
            case "or":
                return StellaBool(lhs.v or rhs.v)
            case "xor":
                return StellaBool(lhs.v != rhs.v)
            case "eq":
                return StellaBool(lhs.v == rhs.v)
            case "ne":
                return StellaBool(lhs.v != rhs.v)
            case _:
                raise ValueError(f"{op} is not a bool binary operation")

    @staticmethod
    def unary_operation(op, operand: StellaBool) -> StellaBool:
        match op:
            case "not":
                return StellaBool(not operand.v)
            case "copy":
                return StellaBool(operand.v)
            case _:
                raise ValueError(f"{op} is not a bool unary operation")


class StellaBinop:
    @staticmethod
    def perform(op, lhs, rhs):
        match lhs, rhs:
            case StellaInt(_), StellaInt(_):
                return StellaInt.binary_operation(op, lhs, rhs)
            case StellaBool(_), StellaBool(_):
                return StellaBool.binary_operation(op, lhs, rhs)
            case _, _:
                raise ValueError(f"Mismathed arguments types for expression {lhs} {op} {rhs}")


class StellaUnop:
    @staticmethod
    def perform(op, arg):
        match arg:
            case StellaInt(_):
                return StellaInt.unary_operation(op, arg)
            case StellaBool(_):
                return StellaBool.unary_operation(op, arg)
            case _:
                raise ValueError(f"Unknowm arguments types for expression {op} {arg}")
