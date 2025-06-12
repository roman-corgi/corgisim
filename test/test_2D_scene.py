# TODO - add tests to verify 2D scene generation, including all priviate methods

from __future__ import annotations
import numpy as np
import astropy.units as u
import pytest

from corgisim.convolution import build_sampling_grid  

@pytest.fixture(scope="module")
def grid_params() -> dict[str, float]:
    """All scalar arguments for build_sampling_grid."""
    return {
        "fine_sampling": 0.25,       # λ/D
        "coarse_sampling": 0.50,
        "iwa": 3.0,
        "owa": 9.0,
        "sampling_theta": 15.0,      # deg
        "resolution_elem": 1.0,
        "res_mas": 50.0,             # toy scale: 1 λ/D = 50 mas
    }

@pytest.fixture(scope="module")
def radial_expect(grid_params) -> np.ndarray:
    """Expected radial grid in λ/D."""
    p = grid_params
    inner = np.arange(0,
                      p["iwa"] + p["fine_sampling"],
                      p["fine_sampling"])
    mid = np.arange(p["iwa"] + p["fine_sampling"],
                    p["owa"],
                    p["coarse_sampling"])
    outer = np.arange(p["owa"],
                      15 + p["resolution_elem"],
                      p["resolution_elem"])
    return np.hstack([inner, mid, outer])


@pytest.fixture(scope="module")
def azimuth_expect(grid_params) -> u.Quantity:
    """Expected azimuth grid (0–360 deg)."""
    step = grid_params["sampling_theta"]
    return np.arange(0, 360, step) * u.deg

def test_build_sampling_grid(
    grid_params,
    radial_expect,
    azimuth_expect,
):
    """Verify grids and unit conversion."""
    out_rad_ld, out_rad_mas, out_az = build_sampling_grid(**grid_params)

    # λ/D grid matches reference
    assert np.allclose(out_rad_ld, radial_expect), "Radial λ/D grid wrong"

    # mas conversion is linear
    scale = grid_params["res_mas"]
    assert np.allclose(out_rad_mas, radial_expect * scale), \
        "λ/D→mas conversion wrong"

    # azimuth grid & unit
    assert out_az.unit is u.deg, "Azimuth grid lacks degree unit"
    assert np.allclose(out_az.value, azimuth_expect.value), \
        "Azimuth grid values wrong"

    # length sanity
    assert len(out_rad_ld) == len(out_rad_mas)
    assert len(out_az) == int(360 / grid_params["sampling_theta"])
