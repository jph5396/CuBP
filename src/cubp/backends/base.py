import numpy as np
from sarpy.io.phase_history.base import CPHDTypeReader

from cubp.args import ImageBounds, Target


def _nan_guard(arr: np.ndarray, message: str):
    assert not np.any(np.isnan(arr)), message


class BaseBackend:
    """
    BaseBackend base class for implementing different backends for
    bpa image formation. Handles some initial set up and orchestration, but
    does not include a complete implementation.
    """

    def __init__(
        self,
        cphd_reader: CPHDTypeReader,
        image_bounds: ImageBounds,
        spacing: float,
        pulse_limit: int = -1,
        target: Target | None = None,
    ):
        self.reader = cphd_reader
        self.spacing = spacing
        self.image_bounds = image_bounds
        self.target = target

        # use all pulses for the image if pulse_limit is set to -1
        if pulse_limit == -1:
            self.pulse_limit: int = self.reader.data_size[0]

        else:
            self.pulse_limit: int = pulse_limit

        self._set_supporting_params()

    def _set_supporting_params(self) -> None:
        """
        Extracts supporting parameters from the reader
        """

        channel_params = self.reader.cphd_meta.Channel.Parameters[0]  # pyright: ignore[reportOptionalMemberAccess]
        self._bandwidth = channel_params.FxBW
        self._fc = channel_params.FxC

        scene_ref = self.reader.cphd_meta.ReferenceGeometry.SRP.ECF
        self._srp_ecf = np.array([scene_ref.X, scene_ref.Y, scene_ref.Z])

        transmitter = self.reader.read_pvp_variable(
            "TxPos",
            index=0,
            the_range=(0, self.pulse_limit),  # pyright: ignore[reportArgumentType]
        )

        receiver = self.reader.read_pvp_variable(
            "RcvPos",
            index=0,
            the_range=(0, self.pulse_limit),  # pyright: ignore[reportArgumentType]
        )

        _nan_guard(
            transmitter,  # pyright: ignore[reportArgumentType]
            "Error: TxPos contains NaNs. Cannot calculate sensor position",
        )

        _nan_guard(
            receiver,  # pyright: ignore[reportArgumentType]
            "Error: RcvPos contains NaNs. Cannot calculate sensor position",
        )

        _nan_guard(self._srp_ecf, "ECEF reference point could not be set appropriately")

        self._src_pos = (transmitter + receiver) / 2  # pyright: ignore[reportOptionalOperand]

    def create_grid(self) -> None:
        """
        Generate the required ECEF grid for bpa.
        """

        raise NotImplementedError

    def form_image(self) -> np.ndarray:
        """
        Run the image formation process.
        """
        raise NotImplementedError
