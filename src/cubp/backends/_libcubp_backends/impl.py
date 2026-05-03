from typing import Any

import numpy as np
from sarpy.io.phase_history.base import CPHDTypeReader

from cubp._libcubp import BPManager, CoordinateGridManager, ECEFCoord, GeodeticCoord
from cubp.args import ImageBounds, Target
from cubp.backends.base import BaseBackend


class LibCuBPBackend(BaseBackend):
    def __init__(
        self,
        cphd_reader: CPHDTypeReader,
        image_bounds: ImageBounds,
        spacing: float,
        pulse_limit: int = -1,
        target: Target | None = None,
    ):
        super().__init__(cphd_reader, image_bounds, spacing, pulse_limit, target)

        # note: this is hacking into sarpy internals to pass the numpy memmap directly.
        # which is valuable because we can read the dat outside of python, but
        # python linters don't like it.
        self.__direct_data: np.ndarray = cphd_reader.data_segment.underlying_array  # type: ignore

        self.__ecef_reference_obj = ECEFCoord(self._srp_ecf[0], self._srp_ecf[1], self._srp_ecf[2])

        self.__target_obj = None
        if target is not None:
            # arguments already confirmed to be set at this point.
            self.__target_obj = GeodeticCoord(
                target.lat,  # pyright: ignore[reportArgumentType]
                target.lon,  # pyright: ignore[reportArgumentType]
                target.alt,  # pyright: ignore[reportArgumentType]
            )

    def create_grid(self) -> None:
        self.ecef_grid = CoordinateGridManager(
            self.image_bounds.x, self.image_bounds.y, self.spacing, self.__ecef_reference_obj, self.__target_obj
        )

        self.ecef_grid.create_grid()

    def form_image(self) -> np.ndarray[tuple[Any, ...], np.dtype[Any]]:
        self.bp_manager = BPManager(
            self.image_bounds.x,
            self.image_bounds.y,
            self.pulse_limit,
            self.reader.data_size[1],  # pyright: ignore
            self._bandwidth,
            self._fc,
            self.__ecef_reference_obj,
            self.ecef_grid,
            self._src_pos,
        )

        for i in range(0, self.pulse_limit):
            pulse: np.ndarray = self.__direct_data[i].astype("<f4")
            self.bp_manager.process_pulse(i, pulse)

        self.bp_manager.finalize_image()

        return self.bp_manager.image_to_numpy()
