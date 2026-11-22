import sys
import traceback
from typing import TextIO
from optimiser.ir.expressions import Form, Unit
from optimiser.ir.json import JsonParser, JsonEmitter
from optimiser.ir.optimisations.naive import UnreachableBlocksElimination
from optimiser.ir.sir import SirEmitter
from optimiser.ssa.optimisations.dce import DCE
from optimiser.ssa.optimisations.naive import NaiveAliasPropagation
from optimiser.ssa.optimisations.sccp import SCCP
from optimiser.ssa.optimisations.cse import CSE
from optimiser.ssa.optimisations.composition import compose
from optimiser.ssa.ssa import SSABuilder, SSALeave


class Source:
    @staticmethod
    def read_from(parser, f: TextIO = sys.stdin) -> Unit:
        return parser.from_string(f.read())

    @staticmethod
    def dump_to(unit: Unit, emiter, f: TextIO = sys.stdout) -> Unit:
        return print(emiter.to_string(unit), file=f)


class Units:
    @staticmethod
    def cfg_to_ssa(unit: Unit) -> Unit:
        if unit.form != "cfg":
            raise ValueError(f"cfg_to_ssa expected 'cfg' format, got '{unit.form}'")

        return Unit(
            format="stella-ir",
            version="1",
            profile=unit.profile,
            form="ssa",
            exports=unit.exports,
            layouts=unit.layouts,
            frames=unit.frames,
            callables=[SSABuilder(callable).ssa() for callable in unit.callables],
        )

    @staticmethod
    def ssa_optimisation(unit: Unit, optimisation) -> Unit:
        if unit.form != "ssa":
            raise ValueError(
                f"ssa optimisation {optimisation.NAME} requires 'ssa' format, got '{unit.form}'"
            )
        return Units.optimisation(unit, "ssa", optimisation)

    @staticmethod
    def optimisation(
        unit: Unit,
        resulting_form: Form,
        optimisation,
    ) -> Unit:
        return Unit(
            format="stella-ir",
            version="1",
            profile=unit.profile,
            form=resulting_form,
            exports=unit.exports,
            layouts=unit.layouts,
            frames=unit.frames,
            callables=[optimisation(callable).optimise() for callable in unit.callables],
        )

    @staticmethod
    def ssa_to_cfg(unit: Unit) -> Unit:
        if unit.form != "ssa":
            raise ValueError(f"ssa_to_cfg expected 'ssa' format, got '{unit.form}'")

        return Unit(
            format="stella-ir",
            version="1",
            profile=unit.profile,
            form="cfg",
            exports=unit.exports,
            layouts=unit.layouts,
            frames=unit.frames,
            callables=[SSALeave(callable).cfg() for callable in unit.callables],
        )


def main(mode: str) -> int:
    try:
        source = Source.read_from(JsonParser, sys.stdin)
        source = Units.optimisation(source, source.form, UnreachableBlocksElimination)
        match mode:
            case "ssa":
                result = Units.cfg_to_ssa(source)
            case "sccp":
                pipeline = compose(SCCP, NaiveAliasPropagation)
                result = Units.ssa_optimisation(source, pipeline)
            case "dce":
                result = Units.ssa_optimisation(source, DCE)
            case "cse":
                result = Units.ssa_optimisation(source, CSE)
            case "lower-cfg":
                result = Units.ssa_to_cfg(source)
            case "pipeline-cfg":
                ssa = Units.cfg_to_ssa(source)
                pipeline = compose(SCCP, NaiveAliasPropagation, CSE, DCE)
                ssa = Units.ssa_optimisation(ssa, pipeline)
                result = Units.ssa_to_cfg(ssa)
            case _:
                raise KeyError(f"no mode {mode} supported")
        Source.dump_to(result, JsonEmitter, sys.stdout)
        return 0
    except Exception as e:
        print(e, file=sys.stderr)
        print("=== DEBUG INFO ===", file=sys.stderr)
        traceback.print_exception(e, file=sys.stderr)
        return 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            f"USAGE: python3 -m {sys.argv[0] if len(sys.argv) > 0 else '<module-name>'} MODE",
            file=sys.stderr,
        )
        exit(1)

    mode = sys.argv[1]
    exit(main(mode))
