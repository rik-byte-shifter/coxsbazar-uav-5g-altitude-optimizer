"""
UAV Altitude Optimization for 5G Coverage in Coastal Terrain
=============================================================
Based on 3GPP TR 36.777 aerial UMa channel model.
Target: Cox's Bazar coastal terrain, Bangladesh.
"""

import numpy as np
from scipy.optimize import minimize_scalar
import warnings

from population_data import load_population_weights

warnings.filterwarnings("ignore")


# ============================================================
# 1. SYSTEM PARAMETERS (3GPP TR 36.777 compliant)
# ============================================================

class SystemParams:
    """All system-level constants. Change here to explore scenarios."""

    fc_sub6 = 3.5e9
    fc_mmwave = 28e9
    c = 3e8

    P_tx_dBm = 27
    P_tx_W = 10 ** ((P_tx_dBm - 30) / 10)

    kT_dBm_Hz = -174
    NF_dB = 7
    BW_MHz = 20
    noise_dBm = kT_dBm_Hz + NF_dB + 10 * np.log10(BW_MHz * 1e6)

    G_tx_dBi = 0
    G_rx_dBi = 0

    SINR_threshold_dB = 0        # Control channel (PDCCH)
    SINR_threshold_data_dB = 3   # Data channel (PDSCH) minimum
    soft_coverage_beta = 1.5
    coverage_saturation_threshold = 0.995

    ISD_m = 1000
    battery_capacity_Wh = 500    # Realistic tethered/aerostat buffer
    hover_power_W = 120          # Power draw at hover
    max_flight_time_min = 45     # Endurance limit

    h_min = 20
    h_max = 500
    h_steps = 250

    grid_size = 40
    grid_extent = 5000

    lat_center = 21.43
    lon_center = 91.98

    def max_endurance_altitude(self) -> float:
        """Returns altitude limit based on battery/energy model."""
        if self.hover_power_W <= 0 or self.battery_capacity_Wh <= 0:
            return self.h_max
        time_available_min = (self.battery_capacity_Wh / self.hover_power_W) * 60
        return min(self.h_max, 300.0) if time_available_min >= self.max_flight_time_min else 150.0


# ============================================================
# 2. 3GPP TR 36.777 AERIAL UMa PATH LOSS MODEL (vectorized)
# ============================================================

class AerialUMaPathLoss:
    """
    3GPP TR 36.777 / 38.901 UMa-style aerial path loss.
    Reference: TR 36.777 V15.0.0 Section 5.2; LoS prob. UMa suburban.
    """

    def __init__(self, fc_Hz: float):
        self.fc = fc_Hz
        self.lam = 3e8 / fc_Hz
        self.fc_GHz = fc_Hz / 1e9

    def los_probability(self, h: float, d_2d) -> np.ndarray:
        """
        LoS probability P_LoS(h, d_2D).
        Ground term: 3GPP 38.901 UMa (d1=18 m, C=300 suburban).
        Aerial correction for h > 22.5 m (TR 36.777 aerial regime).
        """
        d = np.asarray(d_2d, dtype=float)
        scalar = d.ndim == 0
        if scalar:
            d = np.atleast_1d(d)

        C = 300.0
        d1 = 18.0
        p_ground = np.where(
            d < 1.0,
            1.0,
            np.where(
                d <= d1,
                1.0,
                (d1 / d) + np.exp(-d / C) * (1.0 - d1 / d),
            ),
        )

        if h <= 22.5:
            p = p_ground
        else:
            f_h = np.minimum((h - 22.5) / 100.0, 1.0)
            p = 1.0 - (1.0 - p_ground) * np.exp(-f_h * 3.0)
            p = np.minimum(p, 1.0)

        return float(p[0]) if scalar else p

    def pathloss_los(self, h: float, d_2d) -> np.ndarray:
        d = np.asarray(d_2d, dtype=float)
        scalar = d.ndim == 0
        if scalar:
            d = np.atleast_1d(d)

        d_3d = np.sqrt(h ** 2 + d ** 2)
        d_3d = np.maximum(d_3d, 1.0)

        h_E = 1.0
        h_UT = 1.5
        h_BS = h
        d_BP = 4 * (h_BS - h_E) * (h_UT - h_E) * self.fc / 3e8
        d_bp = np.maximum(d_BP, 10.0)

        pl_near = 28.0 + 22 * np.log10(d_3d) + 20 * np.log10(self.fc_GHz)
        pl_far = (
            28.0
            + 40 * np.log10(d_3d)
            + 20 * np.log10(self.fc_GHz)
            - 9 * np.log10(d_bp ** 2 + (h_BS - h_UT) ** 2)
        )
        pl = np.where(d_3d < d_bp, pl_near, pl_far)
        return float(pl[0]) if scalar else pl

    def pathloss_nlos(self, h: float, d_2d) -> np.ndarray:
        d = np.asarray(d_2d, dtype=float)
        scalar = d.ndim == 0
        if scalar:
            d = np.atleast_1d(d)

        d_3d = np.sqrt(h ** 2 + d ** 2)
        d_3d = np.maximum(d_3d, 1.0)
        h_UT = 1.5

        pl_nlos = (
            13.54
            + 39.08 * np.log10(d_3d)
            + 20 * np.log10(self.fc_GHz)
            - 0.6 * (h_UT - 1.5)
        )
        pl_los = self.pathloss_los(h, d)
        pl = np.maximum(pl_nlos, pl_los)
        return float(pl[0]) if scalar else pl

    def mean_pathloss(self, h: float, d_2d) -> np.ndarray:
        p = self.los_probability(h, d_2d)
        pl_los = self.pathloss_los(h, d_2d)
        pl_nlos = self.pathloss_nlos(h, d_2d)

        gain_los = 10 ** (-pl_los / 10)
        gain_nlos = 10 ** (-pl_nlos / 10)
        gain_avg = p * gain_los + (1.0 - p) * gain_nlos
        return -10 * np.log10(gain_avg + 1e-30)


# ============================================================
# 3. RECEIVED SINR CALCULATOR
# ============================================================

class SINRCalculator:
    """Computes SINR at ground points for UAV altitude h."""

    def __init__(self, params: SystemParams, pl_model: AerialUMaPathLoss):
        self.p = params
        self.pl = pl_model

    def interference_dBm(self, h: float, d_2d: np.ndarray | None = None) -> float | np.ndarray:
        I_0 = self.p.noise_dBm - 10  # Base noise/interference floor
        # Height-dependent aerial interference (3GPP TR 36.777 Sec 5.2.2)
        h_dependent = 2.5 * np.log10(max(h, 20.0))
        # Constant neighbor-cell interference (simplified ISD layout)
        I_neighbor = 3.0  # dB margin for 6 surrounding cells
        return I_0 + h_dependent + I_neighbor

    def sinr_grid(self, h: float, d_2d: np.ndarray) -> np.ndarray:
        """SINR [dB] for all ground distances at fixed h."""
        pl = self.pl.mean_pathloss(h, d_2d)
        prx = self.p.P_tx_dBm + self.p.G_tx_dBi + self.p.G_rx_dBi - pl

        s_lin = 10 ** (prx / 10)
        n_lin = 10 ** (self.p.noise_dBm / 10)
        i_lin = 10 ** (self.interference_dBm(h) / 10)
        sinr_lin = s_lin / (n_lin + i_lin)
        return 10 * np.log10(sinr_lin + 1e-30)

    def sinr_dB(self, h: float, d_2d: float) -> float:
        return float(self.sinr_grid(h, np.asarray(d_2d)))


# ============================================================
# 4. COVERAGE METRIC (population-weighted)
# ============================================================

class CoverageMetric:
    """
    Population-weighted coverage at altitude h.
    Binary, soft (sigmoid), and mean SINR metrics.
    """

    def __init__(self, params: SystemParams, sinr_calc: SINRCalculator):
        self.p = params
        self.sinr_calc = sinr_calc
        self.pop_source = "unknown"
        self._build_grid()

    def _build_grid(self):
        n = self.p.grid_size
        ext = self.p.grid_extent

        x = np.linspace(-ext, ext, n)
        y = np.linspace(-ext, ext, n)
        self.X, self.Y = np.meshgrid(x, y)
        self.D_2D = np.sqrt(self.X ** 2 + self.Y ** 2)

        self.W, self.pop_source = load_population_weights(
            self.X, self.Y, self.p
        )
        self.W_total = self.W.sum()

    def _sinr_at_h(self, h: float) -> np.ndarray:
        return self.sinr_calc.sinr_grid(h, self.D_2D)

    def coverage_at_altitude(self, h: float) -> float:
        th = self.p.SINR_threshold_dB
        sinr = self._sinr_at_h(h)
        covered = (sinr >= th).astype(float)
        return float((covered * self.W).sum() / self.W_total)

    def coverage_data_at_altitude(self, h: float) -> float:
        th = self.p.SINR_threshold_data_dB
        sinr = self._sinr_at_h(h)
        covered = (sinr >= th).astype(float)
        return float((covered * self.W).sum() / self.W_total)

    def soft_coverage_at_altitude(self, h: float) -> float:
        th = self.p.SINR_threshold_dB
        beta = self.p.soft_coverage_beta
        sinr = self._sinr_at_h(h)
        soft = 1.0 / (1.0 + np.exp(-beta * (sinr - th)))
        return float((soft * self.W).sum() / self.W_total)

    def soft_coverage_data_at_altitude(self, h: float) -> float:
        th = self.p.SINR_threshold_data_dB
        beta = self.p.soft_coverage_beta
        sinr = self._sinr_at_h(h)
        soft = 1.0 / (1.0 + np.exp(-beta * (sinr - th)))
        return float((soft * self.W).sum() / self.W_total)

    def mean_sinr_weighted(self, h: float) -> float:
        sinr = self._sinr_at_h(h)
        return float((sinr * self.W).sum() / self.W_total)

    def sinr_surface_at_altitude(self, h: float) -> np.ndarray:
        return self._sinr_at_h(h)

    def sweep_metrics(self, h_values: np.ndarray, include_data: bool = False):
        binary = np.empty(len(h_values))
        soft = np.empty(len(h_values))
        mean_sinr = np.empty(len(h_values))
        data_binary = np.empty(len(h_values)) if include_data else None
        th_ctrl = self.p.SINR_threshold_dB
        th_data = self.p.SINR_threshold_data_dB
        beta = self.p.soft_coverage_beta
        for i, h in enumerate(h_values):
            sinr = self._sinr_at_h(h)
            binary[i] = ((sinr >= th_ctrl).astype(float) * self.W).sum() / self.W_total
            soft[i] = (
                (1.0 / (1.0 + np.exp(-beta * (sinr - th_ctrl)))) * self.W
            ).sum() / self.W_total
            mean_sinr[i] = (sinr * self.W).sum() / self.W_total
            if include_data:
                data_binary[i] = (
                    (sinr >= th_data).astype(float) * self.W
                ).sum() / self.W_total
        if include_data:
            return binary, soft, mean_sinr, data_binary
        return binary, soft, mean_sinr

    def sweep_data_metrics(self, h_values: np.ndarray):
        """Sweep using data-channel thresholds (for data-threshold sensitivity)."""
        data_binary = np.empty(len(h_values))
        data_soft = np.empty(len(h_values))
        mean_sinr = np.empty(len(h_values))
        th = self.p.SINR_threshold_data_dB
        beta = self.p.soft_coverage_beta
        for i, h in enumerate(h_values):
            sinr = self._sinr_at_h(h)
            data_binary[i] = ((sinr >= th).astype(float) * self.W).sum() / self.W_total
            data_soft[i] = (
                (1.0 / (1.0 + np.exp(-beta * (sinr - th)))) * self.W
            ).sum() / self.W_total
            mean_sinr[i] = (sinr * self.W).sum() / self.W_total
        return data_binary, data_soft, mean_sinr


# ============================================================
# 5. OPTIMAL ALTITUDE FINDER
# ============================================================

class OptimalAltitudeFinder:
    """Finds h* using soft coverage with plateau fallbacks."""

    def __init__(self, coverage_metric: CoverageMetric, params: SystemParams):
        self.cm = coverage_metric
        self.p = params

    def sweep(self, h_values: np.ndarray | None = None, include_data: bool = False):
        if h_values is None:
            h_values = np.linspace(self.p.h_min, self.p.h_max, self.p.h_steps)
        result = self.cm.sweep_metrics(h_values, include_data=include_data)
        if include_data:
            binary, soft, mean_sinr, data_binary = result
            return h_values, binary, soft, mean_sinr, data_binary
        binary, soft, mean_sinr = result
        return h_values, binary, soft, mean_sinr

    def sweep_data(self, h_values: np.ndarray | None = None):
        if h_values is None:
            h_values = np.linspace(self.p.h_min, self.p.h_max, self.p.h_steps)
        data_binary, data_soft, mean_sinr = self.cm.sweep_data_metrics(h_values)
        return h_values, data_binary, data_soft, mean_sinr

    @staticmethod
    def _argmax_interior(h_values: np.ndarray, metric: np.ndarray) -> tuple:
        idx = int(np.argmax(metric))
        h_star = float(h_values[idx])
        # Prefer interior peak if edge is within 1% of max
        peak = metric[idx]
        margin = 0.01 * peak if peak > 0 else 0.0
        interior = np.where(
            (h_values > h_values[0] + 1)
            & (h_values < h_values[-1] - 1)
            & (metric >= peak - margin)
        )[0]
        if len(interior) > 0:
            mid = interior[len(interior) // 2]
            return float(h_values[mid]), float(metric[mid])
        return h_star, float(peak)

    def _optimize_scalar(self, objective_fn, h_values: np.ndarray, metric: np.ndarray):
        h_sweep, _ = self._argmax_interior(h_values, metric)
        try:
            result = minimize_scalar(
                lambda h: -objective_fn(h),
                bounds=(self.p.h_min, self.p.h_max),
                method="bounded",
                options={"xatol": 1.0},
            )
            h_opt = float(result.x)
        except Exception:
            return h_sweep

        if abs(h_opt - h_sweep) > 15:
            return h_sweep
        return h_opt

    def find_optimal(
        self, h_values: np.ndarray, coverages_binary: np.ndarray,
        coverages_soft: np.ndarray, mean_sinr: np.ndarray,
    ) -> dict:
        sat = self.p.coverage_saturation_threshold
        soft_range = coverages_soft.max() - coverages_soft.min()
        binary_saturated = coverages_binary.max() >= sat
        soft_flat = soft_range < 1e-4

        if binary_saturated and soft_flat:
            h_star, val = self._argmax_interior(h_values, mean_sinr)
            objective = "mean_sinr"
            cov_binary = self.cm.coverage_at_altitude(h_star)
            cov_soft = self.cm.soft_coverage_at_altitude(h_star)
        elif binary_saturated:
            h_star = self._optimize_scalar(
                self.cm.soft_coverage_at_altitude, h_values, coverages_soft
            )
            objective = "soft_coverage"
            cov_binary = self.cm.coverage_at_altitude(h_star)
            cov_soft = self.cm.soft_coverage_at_altitude(h_star)
        else:
            h_star = self._optimize_scalar(
                self.cm.soft_coverage_at_altitude, h_values, coverages_soft
            )
            objective = "soft_coverage"
            cov_binary = self.cm.coverage_at_altitude(h_star)
            cov_soft = self.cm.soft_coverage_at_altitude(h_star)

        h_sweep, cov_sweep = self._argmax_interior(
            h_values,
            mean_sinr if objective == "mean_sinr" else coverages_soft,
        )
        if abs(h_star - h_sweep) > 10:
            h_star = h_sweep
            cov_binary = self.cm.coverage_at_altitude(h_star)
            cov_soft = self.cm.soft_coverage_at_altitude(h_star)

        return {
            "h_star": h_star,
            "cov_star_binary": cov_binary,
            "cov_star_soft": cov_soft,
            "mean_sinr_at_h_star": self.cm.mean_sinr_weighted(h_star),
            "objective_used": objective,
            "h_star_sweep": h_sweep,
            "cov_star_sweep_soft": cov_sweep,
        }

    def find_optimal_data(
        self, h_values: np.ndarray, coverages_data: np.ndarray,
        coverages_soft_data: np.ndarray, mean_sinr: np.ndarray,
    ) -> dict:
        """Find h* optimizing data-channel soft coverage."""
        sat = self.p.coverage_saturation_threshold
        soft_range = coverages_soft_data.max() - coverages_soft_data.min()
        binary_saturated = coverages_data.max() >= sat
        soft_flat = soft_range < 1e-4

        if binary_saturated and soft_flat:
            h_star, val = self._argmax_interior(h_values, mean_sinr)
            objective = "mean_sinr"
            cov_data = self.cm.coverage_data_at_altitude(h_star)
            cov_soft = self.cm.soft_coverage_data_at_altitude(h_star)
        elif binary_saturated:
            h_star = self._optimize_scalar(
                self.cm.soft_coverage_data_at_altitude, h_values, coverages_soft_data
            )
            objective = "soft_coverage_data"
            cov_data = self.cm.coverage_data_at_altitude(h_star)
            cov_soft = self.cm.soft_coverage_data_at_altitude(h_star)
        else:
            h_star = self._optimize_scalar(
                self.cm.soft_coverage_data_at_altitude, h_values, coverages_soft_data
            )
            objective = "soft_coverage_data"
            cov_data = self.cm.coverage_data_at_altitude(h_star)
            cov_soft = self.cm.soft_coverage_data_at_altitude(h_star)

        h_sweep, cov_sweep = self._argmax_interior(
            h_values,
            mean_sinr if objective == "mean_sinr" else coverages_soft_data,
        )
        if abs(h_star - h_sweep) > 10:
            h_star = h_sweep
            cov_data = self.cm.coverage_data_at_altitude(h_star)
            cov_soft = self.cm.soft_coverage_data_at_altitude(h_star)

        return {
            "h_star": h_star,
            "cov_star_data": cov_data,
            "cov_star_control": self.cm.coverage_at_altitude(h_star),
            "cov_star_soft_data": cov_soft,
            "mean_sinr_at_h_star": self.cm.mean_sinr_weighted(h_star),
            "objective_used": objective,
            "h_star_sweep": h_sweep,
            "cov_star_sweep_soft": cov_sweep,
        }

    def soft_coverage_derivative(self, h: float, delta: float = 5.0) -> float:
        c_plus = self.cm.soft_coverage_at_altitude(h + delta)
        c_minus = self.cm.soft_coverage_at_altitude(h - delta)
        return (c_plus - c_minus) / (2 * delta)

    # Backward-compatible aliases
    def find_optimal_numerical(self, h_values, coverages):
        idx = np.argmax(coverages)
        return h_values[idx], coverages[idx]

    def find_optimal_analytical(self):
        h_values = np.linspace(self.p.h_min, self.p.h_max, self.p.h_steps)
        b, s, m = self.cm.sweep_metrics(h_values)
        return self.find_optimal(h_values, b, s, m)["h_star"]

    def coverage_derivative(self, h: float, delta: float = 5.0) -> float:
        return self.soft_coverage_derivative(h, delta)
