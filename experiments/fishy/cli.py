"""Concise command-line errors; library methods retain their typed exceptions."""

import sys
from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

from pydantic import ValidationError

P = ParamSpec("P")
T = TypeVar("T")


class CLI:
    """Translate expected input failures only at executable entrypoints."""

    @staticmethod
    def command(operation: Callable[P, T]) -> Callable[P, T]:
        @wraps(operation)
        def execute(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                return operation(*args, **kwargs)
            except ValidationError as error:
                # Locations and explanations help users repair an input without
                # dumping Pydantic internals or entire supplied records.
                problems = error.errors(include_url=False, include_input=False)
                messages = [
                    f"{'.'.join(str(part) for part in row['loc']) or 'document'}: {row['msg']}"
                    for row in problems[:5]
                ]
                if len(problems) > 5:
                    messages.append(f"and {len(problems) - 5} more validation errors")
                print("Invalid input: " + "; ".join(messages), file=sys.stderr)
                raise SystemExit(2) from None
            except (ValueError, OSError) as error:
                print(f"Input error: {error}", file=sys.stderr)
                raise SystemExit(2) from None

        return execute
