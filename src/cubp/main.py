import numpy as np
from sarpy.io.phase_history.cphd import CPHDTypeReader

from cubp.args import CuBPArguments, ImageBounds, Target
from cubp.backends.numpy.coordinates import build_enu_image_grid, convert_enu_grid_to_ecef


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


def main() -> None:
    args = CuBPArguments()  # pyright: ignore[reportCallIssue]
    print(args)
