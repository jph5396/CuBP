"""
CuBP: cuda backprojection library
"""

from __future__ import annotations

import typing

import numpy
import numpy.typing

__all__: list[str] = [
    "CoordinateGridManager",
    "ECEFCoord",
    "ENUCoord",
    "ENUMatrixTerms",
    "GeodeticCoord",
    "ecef_to_enu",
    "ecef_to_geodetic",
    "geodetic_to_ecef",
]

class CoordinateGridManager:
    def __init__(
        self,
        x_size: typing.SupportsInt | typing.SupportsIndex,
        y_size: typing.SupportsInt | typing.SupportsIndex,
        spacing: typing.SupportsFloat | typing.SupportsIndex,
        reference_point: ECEFCoord,
        target_point: GeodeticCoord | None = None,
    ) -> None: ...
    def create_grid(self) -> None:
        """
        Allocate device memory and run the GPU kernel to build the ECEF coordinate grid
        """
    def grid_to_numpy(self) -> numpy.typing.NDArray[numpy.float64]:
        """
        Copy the device grid to a (x_size*y_size, 3) numpy array (primarily for testing)
        """

class ECEFCoord:
    def __init__(
        self,
        arg0: typing.SupportsFloat | typing.SupportsIndex,
        arg1: typing.SupportsFloat | typing.SupportsIndex,
        arg2: typing.SupportsFloat | typing.SupportsIndex,
    ) -> None: ...
    @property
    def x(self) -> float: ...
    @x.setter
    def x(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None: ...
    @property
    def y(self) -> float: ...
    @y.setter
    def y(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None: ...
    @property
    def z(self) -> float: ...
    @z.setter
    def z(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None: ...

class ENUCoord:
    def __init__(
        self,
        arg0: typing.SupportsFloat | typing.SupportsIndex,
        arg1: typing.SupportsFloat | typing.SupportsIndex,
        arg2: typing.SupportsFloat | typing.SupportsIndex,
    ) -> None: ...
    @property
    def e(self) -> float: ...
    @e.setter
    def e(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None: ...
    @property
    def n(self) -> float: ...
    @n.setter
    def n(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None: ...
    @property
    def u(self) -> float: ...
    @u.setter
    def u(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None: ...

class ENUMatrixTerms:
    def __init__(
        self, lat_rad: typing.SupportsFloat | typing.SupportsIndex, lon_rad: typing.SupportsFloat | typing.SupportsIndex
    ) -> None: ...
    @property
    def cla(self) -> float: ...
    @property
    def cla_clo(self) -> float: ...
    @property
    def cla_slo(self) -> float: ...
    @property
    def clo(self) -> float: ...
    @property
    def sla(self) -> float: ...
    @property
    def sla_clo(self) -> float: ...
    @property
    def sla_slo(self) -> float: ...
    @property
    def slo(self) -> float: ...

class GeodeticCoord:
    def __init__(
        self,
        arg0: typing.SupportsFloat | typing.SupportsIndex,
        arg1: typing.SupportsFloat | typing.SupportsIndex,
        arg2: typing.SupportsFloat | typing.SupportsIndex,
    ) -> None: ...
    @property
    def alt(self) -> float: ...
    @alt.setter
    def alt(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None: ...
    @property
    def lat(self) -> float: ...
    @lat.setter
    def lat(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None: ...
    @property
    def lon(self) -> float: ...
    @lon.setter
    def lon(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None: ...

def ecef_to_enu(arg0: ECEFCoord, arg1: ENUMatrixTerms, arg2: ECEFCoord) -> ENUCoord:
    """
    Converts an ecef coordinate to its enu coordinate
    """

def ecef_to_geodetic(arg0: ECEFCoord) -> GeodeticCoord:
    """
    converts an ECEF coordinate to a geodetic coordinate
    """

def geodetic_to_ecef(arg0: GeodeticCoord) -> ECEFCoord:
    """
    converts a geodetic coordinate to a ECEF coordinate
    """
