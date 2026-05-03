import numpy as np
from sarpy.io.phase_history.base import CPHDTypeReader
from scipy.constants import c
from scipy.fft import fft, fftshift

from cubp.args import ImageBounds, Target
from cubp.backends.base import BaseBackend
from cubp.backends.numpy.coordinates import build_enu_image_grid, convert_enu_grid_to_ecef


def sar_fft(x: np.ndarray) -> np.ndarray:
    """
    perform appropriate FFT shifts for SAR.
    """

    return fftshift(fft(fftshift(x)))


def _prepare_coordinate_grid(
    reader: CPHDTypeReader,
    image_bounds: ImageBounds,
    spacing: float,
    target: Target | None = None,
) -> np.ndarray:
    ref_point = reader.cphd_meta.ReferenceGeometry.SRP.ECF  # pyright: ignore
    ref_vec = np.array([ref_point.X, ref_point.Y, ref_point.Z])

    t = None
    if target is not None:
        t = (target.lat, target.lon, target.alt)

    enu_grid = build_enu_image_grid(image_bounds.x, image_bounds.y, spacing, ref_vec, t)

    ecef_grid = convert_enu_grid_to_ecef(enu_grid, ref_vec)

    return ecef_grid


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

    def create_grid(self):
        self.ecef_grid = _prepare_coordinate_grid(self.reader, self.image_bounds, self.spacing, self.target)

    def form_image(self) -> np.ndarray:
        res_img = np.zeros((self.image_bounds.x * self.image_bounds.y), dtype=np.complex64)

        range_bin_len = self.reader.data_size[1]  # pyright: ignore
        range_res = c / (2 * self._bandwidth)
        range_axis = np.linspace(-range_bin_len / 2, range_bin_len / 2, range_bin_len) * range_res  # pyright: ignore
        k_c = 4 * np.pi * self._fc / c

        for pulse_num in range(1, self.pulse_limit):
            # step 1, get pulse and perform fft.
            pulse_t = sar_fft(self.reader[pulse_num - 1 : pulse_num, :])

            r_center = np.linalg.norm(self._src_pos[pulse_num] - self._srp_ecf)
            r_pixels = np.linalg.norm(self.ecef_grid - self._src_pos[pulse_num], axis=1)
            dr = r_center - r_pixels

            real_value = np.interp(dr, range_axis, pulse_t.real)  # pyright: ignore
            imag_value = np.interp(dr, range_axis, pulse_t.imag)  # pyright: ignore

            signal: np.ndarray = real_value + 1j * imag_value
            phase_correction_term: np.ndarray = np.exp(-1j * k_c * dr)
            res_img += signal * phase_correction_term

        center_source_pos = self._src_pos[self.pulse_limit // 2, :]
        dr = np.linalg.norm(center_source_pos - self._srp_ecf) - np.linalg.norm(
            self.ecef_grid - center_source_pos, axis=1
        )
        final_correction: np.ndarray = np.exp(1j * k_c * dr)

        res_img *= final_correction

        return res_img.reshape(self.image_bounds.x, self.image_bounds.y)
