"""
Sensitivity analysis: h* vs P_tx and SINR threshold (Sub-6 GHz).
Control-channel sweep (Fig 4) and data-channel sweep (Fig 5).
"""

import csv

import numpy as np

from uav_optimizer import (
    SystemParams,
    AerialUMaPathLoss,
    SINRCalculator,
    CoverageMetric,
    OptimalAltitudeFinder,
)
from visualizations import (
    plot_fig4_sensitivity,
    plot_fig5_data_sensitivity,
    OUTPUT_DIR,
)


def run_control_sensitivity(
    p_tx_values=None,
    threshold_values=None,
    h_steps=80,
):
    p_tx_values = p_tx_values or [25, 27, 30]
    threshold_values = threshold_values or [-3, 0, 3]

    results = []
    print("\n  Control-channel sensitivity (Sub-6 GHz)...")

    for p_tx in p_tx_values:
        for th in threshold_values:
            params = SystemParams()
            params.P_tx_dBm = p_tx
            params.SINR_threshold_dB = th
            params.h_steps = h_steps
            params.h_max = min(params.h_max, params.max_endurance_altitude())

            pl = AerialUMaPathLoss(params.fc_sub6)
            sinr = SINRCalculator(params, pl)
            cov = CoverageMetric(params, sinr)
            opt = OptimalAltitudeFinder(cov, params)

            h_values = np.linspace(params.h_min, params.h_max, params.h_steps)
            h_values, b, s, m = opt.sweep(h_values)
            out = opt.find_optimal(h_values, b, s, m)

            row = {
                "P_tx_dBm": p_tx,
                "SINR_threshold_dB": th,
                "h_star": out["h_star"],
                "coverage_binary_pct": out["cov_star_binary"] * 100,
                "coverage_data_pct": cov.coverage_data_at_altitude(out["h_star"]) * 100,
                "coverage_soft_pct": out["cov_star_soft"] * 100,
                "objective": out["objective_used"],
            }
            results.append(row)
            print(
                f"    P_tx={p_tx} dBm, γ_th={th} dB -> h*={row['h_star']:.0f} m "
                f"({row['coverage_binary_pct']:.1f}% control)"
            )

    csv_path = OUTPUT_DIR / "sensitivity_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    fig_path = plot_fig4_sensitivity(results)
    return fig_path, csv_path, results


def run_data_threshold_sensitivity(
    p_tx_dBm=27,
    data_threshold_values=None,
    h_steps=80,
):
    data_threshold_values = data_threshold_values or [0, 3, 6, 9]

    results = []
    print("\n  Data-channel sensitivity (Sub-6 GHz, P_tx=27 dBm)...")

    for th_data in data_threshold_values:
        params = SystemParams()
        params.P_tx_dBm = p_tx_dBm
        params.SINR_threshold_dB = 0
        params.SINR_threshold_data_dB = th_data
        params.h_steps = h_steps
        params.h_max = min(params.h_max, params.max_endurance_altitude())

        pl = AerialUMaPathLoss(params.fc_sub6)
        sinr = SINRCalculator(params, pl)
        cov = CoverageMetric(params, sinr)
        opt = OptimalAltitudeFinder(cov, params)

        h_values = np.linspace(params.h_min, params.h_max, params.h_steps)
        h_values, data_b, data_s, m = opt.sweep_data(h_values)
        out = opt.find_optimal_data(h_values, data_b, data_s, m)

        row = {
            "P_tx_dBm": p_tx_dBm,
            "SINR_threshold_data_dB": th_data,
            "h_star": out["h_star"],
            "coverage_data_pct": out["cov_star_data"] * 100,
            "coverage_control_pct": out["cov_star_control"] * 100,
            "coverage_soft_data_pct": out["cov_star_soft_data"] * 100,
            "objective": out["objective_used"],
        }
        results.append(row)
        print(
            f"    γ_th,data={th_data} dB -> h*={row['h_star']:.0f} m "
            f"({row['coverage_data_pct']:.1f}% data, "
            f"{row['coverage_control_pct']:.1f}% control)"
        )

    csv_path = OUTPUT_DIR / "sensitivity_data_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    fig_path = plot_fig5_data_sensitivity(results)
    return fig_path, csv_path, results


def run(
    p_tx_values=None,
    threshold_values=None,
    data_threshold_values=None,
    h_steps=80,
):
    fig4_path, csv_path, _ = run_control_sensitivity(
        p_tx_values=p_tx_values,
        threshold_values=threshold_values,
        h_steps=h_steps,
    )
    fig5_path, data_csv_path, _ = run_data_threshold_sensitivity(
        data_threshold_values=data_threshold_values,
        h_steps=h_steps,
    )
    return fig4_path, fig5_path, csv_path, data_csv_path


if __name__ == "__main__":
    run()
