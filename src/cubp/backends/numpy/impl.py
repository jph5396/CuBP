from sarpy.io.phase_history.base import CPHDTypeReader

from cubp.args import ImageBounds, Target
from cubp.backends.base import BaseBackend


class NumpyBackend(BaseBackend):
    def __init__(
        self,
        cphd_reader: CPHDTypeReader,
        image_bounds: ImageBounds,
        spacing: float,
        pulse_limit: int = -1,
        target: Target | None = None,
    ):
        super().__init__(cphd_reader, image_bounds, spacing, pulse_limit, target)
