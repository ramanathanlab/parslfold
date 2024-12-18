"""Utility module."""

from __future__ import annotations

import functools
import json
import sys
from pathlib import Path
from typing import Callable
from typing import TypeVar

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel as _BaseModel

if sys.version_info >= (3, 10):
    from typing import ParamSpec
else:
    from typing_extensions import ParamSpec

T = TypeVar('T')
P = ParamSpec('P')


def exception_handler(
    default_return: T | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T | None]]:
    """Handle exceptions in a function by returning a `default_return` value.

    A decorator factory that returns a decorator formatted with the
    default_return that wraps a function.
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T | None]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T | None:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(
                    f'{func.__name__} raised an exception: {e} '
                    f'On input {args}, {kwargs}\nReturning {default_return}',
                )
                return default_return

        return wrapper

    return decorator


class BaseModel(_BaseModel):
    """Provide an easy interface to read/write YAML files."""

    def dump_yaml(self, filename: str | Path) -> None:
        """Dump settings to a YAML file."""
        with open(filename, mode='w') as fp:
            yaml.dump(
                json.loads(self.model_dump_json()),
                fp,
                indent=4,
                sort_keys=False,
            )

    @classmethod
    def from_yaml(cls: type[T], filename: str | Path) -> T:
        """Load settings from a YAML file."""
        with open(filename) as fp:
            raw_data = yaml.safe_load(fp)
        return cls(**raw_data)


def batch_data(data: list[T], chunk_size: int) -> list[list[T]]:
    """Batch data into chunks of size chunk_size.

    Parameters
    ----------
    data : list[T]
        The data to batch.
    chunk_size : int
        The size of each batch.

    Returns
    -------
    list[list[T]]
        The batched data.
    """
    batches = [
        data[i * chunk_size : (i + 1) * chunk_size]
        for i in range(0, len(data) // chunk_size)
    ]
    if len(data) > chunk_size * len(batches):
        batches.append(data[len(batches) * chunk_size :])
    return batches
