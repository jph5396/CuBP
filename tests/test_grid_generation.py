"""
Tests for GPU ECEF grid generation against the numpy/sarpy reference pipeline.
"""

import numpy as np
import pytest
from sarpy.geometry.geocoords import geodetic_to_ecf

from cubp._libcubp import CoordinateGridManager, ECEFCoord, GeodeticCoord
from cubp.backends.numpy.coordinates import build_enu_image_grid, convert_enu_grid_to_ecef

ECEF_TOL_M = 1e-3  # 1 mm

# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

# (lat_deg, lon_deg, alt_m)
_REFS = {
    "equator": (0.0, 0.0, 0.0),
    "london": (51.5074, -0.1278, 15.0),
    "white-house": (38.897957, -77.036560, 17.98),
    "high-lat": (70.0, 45.0, 1000.0),
}

# (x_size, y_size, spacing_m, ref_key, target_geodetic_deg_or_None, case_id)
GRID_CASES = [
    pytest.param(3, 3, 1.0, "equator", None, id="3x3-equator-no-target"),
    pytest.param(3, 3, 1.0, "london", None, id="3x3-london-no-target"),
    pytest.param(5, 5, 10.0, "white-house", None, id="5x5-white-house-no-target"),
    pytest.param(4, 6, 5.0, "high-lat", None, id="4x6-high-lat-asymmetric"),
    pytest.param(10, 10, 100.0, "london", None, id="10x10-london-100m"),
    pytest.param(5, 5, 10.0, "white-house", (38.89, -77.03, 10.0), id="5x5-white-house-with-target"),
    pytest.param(7, 7, 50.0, "london", (51.510, -0.120, 20.0), id="7x7-london-with-target"),
]


def _numpy_ecef_grid(x_size, y_size, spacing, ref_ecef_arr, target_geodetic_arr=None):
    enu = build_enu_image_grid(x_size, y_size, spacing, ref_ecef_arr, target=target_geodetic_arr)
    return convert_enu_grid_to_ecef(enu, ref_ecef_arr)


def _gpu_ecef_grid(x_size, y_size, spacing, ref_ecef_arr, target_geodetic_arr=None):
    ref_ecef = ECEFCoord(ref_ecef_arr[0], ref_ecef_arr[1], ref_ecef_arr[2])

    target = None
    if target_geodetic_arr is not None:
        target = GeodeticCoord(
            target_geodetic_arr[0],
            target_geodetic_arr[1],
            target_geodetic_arr[2],
        )

    mgr = CoordinateGridManager(x_size, y_size, spacing, ref_ecef, target)
    mgr.create_grid()
    return mgr.grid_to_numpy()


@pytest.mark.parametrize("x_size,y_size,spacing,ref_key,target_llh", GRID_CASES)
def test_ecef_grid_matches_numpy(x_size, y_size, spacing, ref_key, target_llh):
    ref_llh = _REFS[ref_key]
    ref_ecef = geodetic_to_ecf(list(ref_llh))

    target_arr = np.array(target_llh) if target_llh is not None else None

    expected = _numpy_ecef_grid(x_size, y_size, spacing, ref_ecef, target_arr)
    actual = _gpu_ecef_grid(x_size, y_size, spacing, ref_ecef, target_arr)

    assert actual.shape == expected.shape, f"Shape mismatch: GPU={actual.shape}, numpy={expected.shape}"
    np.testing.assert_allclose(
        actual,
        expected,
        atol=ECEF_TOL_M,
        err_msg=f"ECEF grid mismatch for {ref_key} {x_size}x{y_size} spacing={spacing}",
    )
