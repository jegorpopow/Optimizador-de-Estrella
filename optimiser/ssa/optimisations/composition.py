from typing import Callable


def compose(*optimisations):
    class Result:
        NAME = f"Composition({' . '.join(opt.NAME for opt in reversed(optimisations))})"

        def __init__(self, function: Callable):
            self.function = function

        def optimise(self) -> Callable:
            result = self.function
            for optimisation in optimisations:
                result = optimisation(result).optimise()
            return result

    return Result
