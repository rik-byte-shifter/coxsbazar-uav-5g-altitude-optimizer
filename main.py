"""
MAIN RUNNER
===========
UAV Altitude Optimization for 5G Coverage in Coastal Terrain

Usage:
    python main.py

Outputs (saved in ./outputs/):
    fig1_sinr_surface_3d.png
    fig2_coverage_vs_altitude.png
    fig3_frequency_comparison.png
    bonus_population_sinr_maps.png
    fig4_sensitivity_hstar.png
    fig5_sensitivity_data_hstar.png
    live_full_project.html
    live_3d_optimizer.html
    live_sinr_optimizer.html
    sensitivity_results.csv
    results_summary.txt
"""

import shutil
import sys
import time
from pathlib import Path

import numpy as np

from uav_optimizer import (
    SystemParams,
    AerialUMaPathLoss,
    SINRCalculator,
    CoverageMetric,
    OptimalAltitudeFinder,
)
from visualizations import (
    plot_fig1_sinr_surface,
    plot_fig2_coverage_curve,
    plot_fig3_frequency_comparison,
    plot_bonus_heatmaps,
)

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_single_frequency(fc_Hz: float, label: str, params_template: SystemParams):
    print(f"\n{'=' * 60}")
    print(f"  Running: {label}  (fc = {fc_Hz / 1e9:.1f} GHz)")
    print(f"{'=' * 60}")

    params_template.h_max = min(
        params_template.h_max, params_template.max_endurance_altitude()
    )
    print(f"  Effective h_max (endurance): {params_template.h_max:.0f} m")

    pl_model = AerialUMaPathLoss(fc_Hz=fc_Hz)
    sinr_calc = SINRCalculator(params_template, pl_model)
    cov_metric = CoverageMetric(params_template, sinr_calc)
    optimizer = OptimalAltitudeFinder(cov_metric, params_template)

    print(f"  Population weights: {cov_metric.pop_source}")

    print("  Step 1: Altitude sweep...")
    t0 = time.time()
    h_values = np.linspace(
        params_template.h_min, params_template.h_max, params_template.h_steps
    )
    h_values, coverages, soft_coverages, mean_sinr, data_coverages = optimizer.sweep(
        h_values, include_data=True
    )
    print(f"    Sweep done in {time.time() - t0:.1f}s")

    opt = optimizer.find_optimal(h_values, coverages, soft_coverages, mean_sinr)
    h_star = opt["h_star"]
    cov_star = opt["cov_star_binary"]
    cov_soft = opt["cov_star_soft"]
    cov_data = cov_metric.coverage_data_at_altitude(h_star)

    print(f"    h* (sweep ref)       = {opt['h_star_sweep']:.1f} m")
    print(f"    h* (optimized)       = {h_star:.1f} m  "
          f"-> control cov = {cov_star * 100:.1f}%")
    print(f"    data coverage at h*  = {cov_data * 100:.1f}%")
    print(f"    soft coverage at h*  = {cov_soft * 100:.1f}%")
    print(f"    mean SINR at h*      = {opt['mean_sinr_at_h_star']:.2f} dB")
    print(f"    objective used       = {opt['objective_used']}")

    dC = optimizer.soft_coverage_derivative(h_star)
    print(f"    d(soft cov)/dh at h* = {dC:.6f}  (should be ~ 0)")

    peak_binary = coverages.max()
    if peak_binary >= params_template.coverage_saturation_threshold:
        print("    NOTE: binary coverage saturated; h* from soft/mean-SINR objective.")

    return {
        "pl_model": pl_model,
        "sinr_calc": sinr_calc,
        "cov_metric": cov_metric,
        "optimizer": optimizer,
        "h_values": h_values,
        "coverages": coverages,
        "soft_coverages": soft_coverages,
        "data_coverages": data_coverages,
        "mean_sinr": mean_sinr,
        "h_star": h_star,
        "cov_star": cov_star,
        "cov_star_data": cov_data,
        "cov_star_soft": cov_soft,
        "opt": opt,
        "label": label,
        "fc_Hz": fc_Hz,
    }


def _check_interior_peak(h_values, metric, name: str) -> bool:
    idx = np.argmax(metric)
    at_low = idx == 0 and metric[0] >= metric.max() - 1e-6
    at_high = idx == len(metric) - 1 and metric[-1] >= metric.max() - 1e-6
    if at_low or at_high:
        print(f"  WARNING: {name} peak at altitude boundary; consider retuning SystemParams.")
        return False
    return True


def run_pipeline():
    """
    Run Sub-6 and mmWave optimization sweeps.
    Returns (res_sub6, res_mm, params_sub6, params_mm, ok_sub6, ok_mm).
    """
    print("\n" + "=" * 60)
    print("  UAV 5G Altitude Optimizer - Cox's Bazar, Bangladesh")
    print("  Standard: 3GPP TR 36.777 Aerial UMa")
    print("=" * 60)

    params_sub6 = SystemParams()
    res_sub6 = run_single_frequency(
        fc_Hz=params_sub6.fc_sub6,
        label="Sub-6 GHz (3.5 GHz)",
        params_template=params_sub6,
    )

    params_mm = SystemParams()
    res_mm = run_single_frequency(
        fc_Hz=params_mm.fc_mmwave,
        label="mmWave (28 GHz)",
        params_template=params_mm,
    )

    ok_sub6 = _check_interior_peak(
        res_sub6["h_values"], res_sub6["soft_coverages"], "Sub-6 soft coverage"
    )
    ok_mm = _check_interior_peak(
        res_mm["h_values"], res_mm["soft_coverages"], "mmWave soft coverage"
    )

    return res_sub6, res_mm, params_sub6, params_mm, ok_sub6, ok_mm


def main():
    res_sub6, res_mm, params_sub6, params_mm, ok_sub6, ok_mm = run_pipeline()

    print("\n" + "=" * 60)
    print("  Generating publication figures...")
    print("=" * 60)

    fig1_path = plot_fig1_sinr_surface(
        res_sub6["cov_metric"],
        res_sub6["h_star"],
        params_sub6,
        res_sub6["sinr_calc"],
    )

    fig2_path = plot_fig2_coverage_curve(
        res_sub6["h_values"],
        res_sub6["coverages"],
        res_sub6["soft_coverages"],
        res_sub6["data_coverages"],
        res_sub6["h_star"],
        res_sub6["cov_star"],
        params_sub6,
    )

    fig3_path = plot_fig3_frequency_comparison(
        res_sub6["h_values"],
        res_sub6["coverages"],
        res_sub6["h_star"],
        res_sub6["cov_star"],
        res_mm["coverages"],
        res_mm["h_star"],
        res_mm["cov_star"],
        params_sub6,
        params_mm,
    )

    bonus_path = plot_bonus_heatmaps(
        res_sub6["cov_metric"],
        res_sub6["h_star"],
        params_sub6,
    )

    fig4_path = None
    fig5_path = None
    try:
        import sensitivity

        fig4_path, fig5_path, _, _ = sensitivity.run()
    except Exception as exc:
        print(f"  Sensitivity analysis skipped: {exc}")

    try:
        import live_viz

        live_path = live_viz.generate(res_sub6, res_mm)
        if live_path:
            print(f"  Open LIVE FULL simulation: {live_path}")
    except Exception as exc:
        print(f"  Live visualization skipped: {exc}")

    delta_h = res_mm["h_star"] - res_sub6["h_star"]
    pop_src = res_sub6["cov_metric"].pop_source

    summary = f"""UAV Altitude Optimization - Results Summary
============================================
Standard  : 3GPP TR 36.777 Aerial UMa
Location  : Cox's Bazar, Bangladesh (coastal flat terrain)
Population: {pop_src}
Grid      : {params_sub6.grid_size}x{params_sub6.grid_size} points,
            +/-{params_sub6.grid_extent} m extent
SINR threshold (control): {params_sub6.SINR_threshold_dB} dB
SINR threshold (data)   : {params_sub6.SINR_threshold_data_dB} dB

--- Sub-6 GHz (fc = {params_sub6.fc_sub6 / 1e9:.1f} GHz) ---
  Optimal altitude h*   = {res_sub6['h_star']:.1f} m
  Control coverage      = {res_sub6['cov_star'] * 100:.2f}%
  Data coverage         = {res_sub6['cov_star_data'] * 100:.2f}%
  Soft coverage         = {res_sub6['cov_star_soft'] * 100:.2f}%
  Mean SINR at h*       = {res_sub6['opt']['mean_sinr_at_h_star']:.2f} dB
  Objective             = {res_sub6['opt']['objective_used']}
  Tx power              = {params_sub6.P_tx_dBm} dBm
  Bandwidth             = {params_sub6.BW_MHz} MHz

--- mmWave (fc = {params_mm.fc_mmwave / 1e9:.0f} GHz) ---
  Optimal altitude h*   = {res_mm['h_star']:.1f} m
  Control coverage      = {res_mm['cov_star'] * 100:.2f}%
  Data coverage         = {res_mm['cov_star_data'] * 100:.2f}%
  Soft coverage         = {res_mm['cov_star_soft'] * 100:.2f}%
  Mean SINR at h*       = {res_mm['opt']['mean_sinr_at_h_star']:.2f} dB
  Objective             = {res_mm['opt']['objective_used']}
  Tx power              = {params_mm.P_tx_dBm} dBm
  Bandwidth             = {params_mm.BW_MHz} MHz

--- Key Finding ---
  Delta h* (mmWave - Sub6) = {delta_h:.1f} m

--- Output Files ---
  {fig1_path}
  {fig2_path}
  {fig3_path}
  {bonus_path}
  {fig4_path or '(control sensitivity skipped)'}
  {fig5_path or '(data sensitivity skipped)'}
  outputs/live_full_project.html
  outputs/live_3d_optimizer.html
  outputs/live_ground_sinr.html
"""
    print(summary)
    with open(OUTPUT_DIR / "results_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)

    project_root = Path(__file__).parent
    for png in OUTPUT_DIR.glob("*.png"):
        shutil.copy2(png, project_root / png.name)
    print(f"  Copied {len(list(OUTPUT_DIR.glob('*.png')))} PNG(s) to project root for LaTeX.")

    print("\n  All done! Check the 'outputs/' folder for figures.")

    if not (ok_sub6 and ok_mm):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
