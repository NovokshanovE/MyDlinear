from collections import UserDict
from dataclasses import dataclass
from operator import attrgetter
from typing import Any, Callable, Dict, Type, Tuple

from toolz import valmap


@dataclass
class Input:
    shape: Tuple[int, ...]
    dtype: Any
    required: bool = True



@dataclass
class InputSpec(UserDict):
    data: Dict[str, Input]
    zeros_fn: Callable

    @property
    def shapes(self) -> Dict[str, Tuple[int, ...]]:
        return valmap(attrgetter("shape"), self)

    @property
    def dtypes(self) -> Dict[str, Type]:
        return valmap(attrgetter("dtype"), self)

    def zeros(self):
        return {
            name: self.zeros_fn(input.shape, dtype=input.dtype)
            for name, input in self.items()
        }