"""Tests for UAV altitude optimizer."""

import numpy as np
import pytest

from uav_optimizer import (
    SystemParams,
    AerialUMaPathLoss,
    SINRCalculator,
    CoverageMetric,
    OptimalAltitudeFinder,
)


def _pipeline(fc: float, params: SystemParams | None = None):
    params = params or SystemParams()
    params.h_max = min(params.h_max, params.max_endurance_altitude())
    pl = AerialUMaPathLoss(fc)
    sinr = SINRCalculator(params, pl)
    cov = CoverageMetric(params, sinr)
    opt = OptimalAltitudeFinder(cov, params)
    h_values = np.linspace(params.h_min, params.h_max, min(params.h_steps, 100))
    h_values, b, s, m = opt.sweep(h_values)
    out = opt.find_optimal(h_values, b, s, m)
    return h_values, b, s, m, out, opt


def test_pathloss_increases_with_distance():
    pl = AerialUMaPathLoss(SystemParams.fc_sub6)
    h = 100.0
    d_near = np.array([50.0, 100.0])
    d_far = np.array([2000.0, 4000.0])
    pl_near = pl.mean_pathloss(h, d_near)
    pl_far = pl.mean_pathloss(h, d_far)
    assert np.all(pl_far > pl_near)


def test_sinr_grid_shape():
    params = SystemParams()
    pl = AerialUMaPathLoss(params.fc_sub6)
    sinr = SINRCalculator(params, pl)
    d = np.linspace(10, 1000, 20)
    grid = sinr.sinr_grid(80.0, d)
    assert grid.shape == (20,)
    assert np.all(np.isfinite(grid))


def test_coverage_has_interior_peak_sub6():
    h_values, _, soft, _, out, _ = _pipeline(SystemParams.fc_sub6)
    idx = np.argmax(soft)
    assert 0 < idx < len(h_values) - 1, "Sub-6 soft coverage should peak inside altitude range"
    assert out["h_star"] >= h_values[0] and out["h_star"] <= h_values[-1]


def test_coverage_has_interior_peak_mmwave():
    h_values, _, soft, _, out, _ = _pipeline(SystemParams.fc_mmwave)
    idx = np.argmax(soft)
    assert 0 < idx < len(h_values) - 1, "mmWave soft coverage should peak inside altitude range"


def test_h_star_sweep_scipy_agree():
    h_values, _, soft, _, out, _ = _pipeline(SystemParams.fc_mmwave)
    assert abs(out["h_star"] - out["h_star_sweep"]) <= 15


def test_soft_coverage_derivative_near_zero():
    _, _, _, _, out, opt = _pipeline(SystemParams.fc_mmwave)
    deriv = opt.soft_coverage_derivative(out["h_star"], delta=3.0)
    assert abs(deriv) < 0.05


def test_max_endurance_altitude_default():
    params = SystemParams()
    assert params.max_endurance_altitude() == 300.0


def test_interference_independent_of_distance():
    params = SystemParams()
    pl = AerialUMaPathLoss(params.fc_sub6)
    sinr = SINRCalculator(params, pl)
    i_near = sinr.interference_dBm(100.0, np.array([50.0, 100.0]))
    i_far = sinr.interference_dBm(100.0, np.array([2000.0, 4000.0]))
    assert np.allclose(i_near, i_far)
    assert sinr.interference_dBm(100.0) == sinr.interference_dBm(100.0, np.array([500.0]))
