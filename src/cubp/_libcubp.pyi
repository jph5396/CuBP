"""
CuBP: cuda backprojection library
"""

from __future__ import annotations

import typing

import numpy
import numpy.typing

__all__: list[str] = ["scale", "test"]

def scale(
    arr: typing.Annotated[numpy.typing.ArrayLike, numpy.float32], scalar: typing.SupportsFloat | typing.SupportsIndex
) -> numpy.typing.NDArray[numpy.float32]:
    """
    Element-wise multiply a float32 numpy array by a scalar on the GPU
    """

def test(arr: typing.Annotated[numpy.typing.ArrayLike, numpy.float32]) -> None:
    """
    testing func
    """
