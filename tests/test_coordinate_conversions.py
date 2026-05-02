"""
Tests for libcubp coordinate conversions against sarpy reference implementations.

libcubp conventions:
  - geodetic_to_ecef: lat/lon inputs in degrees
  - ecef_to_geodetic: lat/lon outputs in radians (converted to degrees below for comparison)
"""

import math

import numpy as np
import pytest
from sarpy.geometry.geocoords import ecf_to_geodetic, geodetic_to_ecf

from cubp._libcubp import ECEFCoord, GeodeticCoord, ecef_to_geodetic, geodetic_to_ecef

# (lat_deg, lon_deg, alt_m)
GEODETIC_CASES = [
    pytest.param(0.0, 0.0, 0.0, id="equator-prime-meridian"),
    pytest.param(51.5074, -0.1278, 15.0, id="london"),
    pytest.param(-33.8688, 151.2093, 50.0, id="sydney"),
    pytest.param(70.0, 45.0, 1000.0, id="high-latitude"),
    pytest.param(-89.0, 0.0, 0.0, id="near-south-pole"),
    pytest.param(38.897957, -77.036560, 17.98, id="white-house"),
    pytest.param(37.334606, -122.009102, 72.0, id="apple-park"),
]

# Absolute tolerances
ECEF_TOL_M = 1e-3  # 1 mm
GEOD_TOL_DEG = 1e-8  # sub-microdegree


def _cubp_ecef_to_array(coord: ECEFCoord) -> np.ndarray:
    return np.array([coord.x, coord.y, coord.z])


def _cubp_geodetic_to_degrees(coord: GeodeticCoord) -> np.ndarray:
    """ecef_to_geodetic returns lat/lon in radians; convert to degrees for comparison."""
    return np.array(
        [
            math.degrees(coord.lat),
            math.degrees(coord.lon),
            coord.alt,
        ]
    )


@pytest.mark.parametrize("lat,lon,alt", GEODETIC_CASES)
def test_geodetic_to_ecef(lat, lon, alt):
    sarpy_ecef = geodetic_to_ecf([lat, lon, alt])
    cubp_ecef = _cubp_ecef_to_array(geodetic_to_ecef(GeodeticCoord(lat, lon, alt)))

    np.testing.assert_allclose(
        cubp_ecef, sarpy_ecef, atol=ECEF_TOL_M, err_msg=f"geodetic_to_ecef mismatch for ({lat}, {lon}, {alt})"
    )


@pytest.mark.parametrize("lat,lon,alt", GEODETIC_CASES)
def test_ecef_to_geodetic(lat, lon, alt):
    sarpy_ecef = geodetic_to_ecf([lat, lon, alt])
    sarpy_geod = ecf_to_geodetic(sarpy_ecef)  # [lat_deg, lon_deg, alt_m]

    cubp_ecef = ECEFCoord(sarpy_ecef[0], sarpy_ecef[1], sarpy_ecef[2])
    cubp_geod = _cubp_geodetic_to_degrees(ecef_to_geodetic(cubp_ecef))

    np.testing.assert_allclose(
        cubp_geod, sarpy_geod, atol=GEOD_TOL_DEG, err_msg=f"ecef_to_geodetic mismatch for ({lat}, {lon}, {alt})"
    )
