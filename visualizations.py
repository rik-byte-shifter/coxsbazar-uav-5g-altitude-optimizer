"""
Visualization Module
====================
Publication-ready figures for UAV altitude optimization.
"""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 1.2,
    "grid.alpha": 0.35,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_fig1_sinr_surface(coverage_metric, h_star, params, sinr_calc):
    """3D surface: h vs ground distance vs SINR; h* ridge line."""
    print("  Generating Fig 1: 3D SINR Surface...")

    h_arr = np.linspace(params.h_min, params.h_max, 80)
    d_arr = np.linspace(1, params.grid_extent, 80)
    H, D = np.meshgrid(h_arr, d_arr, indexing="ij")

    Z = np.zeros_like(H, dtype=float)
    for i, h in enumerate(h_arr):
        Z[i, :] = sinr_calc.sinr_grid(h, D[i, :])

    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        H, D / 1000, Z,
        cmap="viridis",
        alpha=0.85,
        linewidth=0,
        antialiased=True,
    )

    d_ridge = np.linspace(1, params.grid_extent, 200)
    z_ridge = sinr_calc.sinr_grid(h_star, d_ridge)
    ax.plot(
        [h_star] * len(d_ridge),
        d_ridge / 1000,
        z_ridge,
        color="red",
        linewidth=3,
        zorder=10,
    )

    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=12, pad=0.1)
    cbar.set_label("SINR (dB)", rotation=270, labelpad=15)

    th = params.SINR_threshold_dB
    H_th, D_th = np.meshgrid(
        [params.h_min, params.h_max],
        [1, params.grid_extent / 1000],
    )
    Z_th = np.full_like(H_th, th, dtype=float)
    ax.plot_surface(H_th, D_th, Z_th, alpha=0.15, color="orange")

    ax.set_xlabel("UAV Altitude h (m)", labelpad=10)
    ax.set_ylabel("Ground Distance d (km)", labelpad=10)
    ax.set_zlabel("SINR (dB)", labelpad=10)
    ax.set_title(
        f"3GPP TR 36.777 Aerial UMa SINR Surface\n"
        f"Cox's Bazar | fc = {params.fc_sub6 / 1e9:.1f} GHz | h* = {h_star:.0f} m",
        pad=15,
    )

    legend_elements = [
        Line2D([0], [0], color="red", lw=3, label=f"h* = {h_star:.0f} m ridge"),
        mpatches.Patch(color="orange", alpha=0.4, label=f"SINR threshold = {th} dB"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)
    ax.view_init(elev=25, azim=-50)

    plt.tight_layout()
    path = OUTPUT_DIR / "fig1_sinr_surface_3d.png"
    plt.savefig(path)
    plt.close()
    print(f"    Saved: {path}")
    return path


def plot_fig2_coverage_curve(
    h_values, coverages, soft_coverages, data_coverages, h_star, cov_star, params
):
    """Control/data coverage vs h; derivative of soft coverage."""
    print("  Generating Fig 2: Coverage vs Altitude...")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    cov_pct = coverages * 100
    soft_pct = soft_coverages * 100
    data_pct = data_coverages * 100
    th_data = params.SINR_threshold_data_dB

    ax1.plot(
        h_values, cov_pct, color="steelblue", lw=2.5,
        label=f"Control coverage (γ_th = {params.SINR_threshold_dB} dB)",
    )
    ax1.plot(
        h_values, data_pct, color="purple", lw=2.0,
        label=f"Data coverage (γ_th,data = {th_data} dB)",
    )
    ax1.plot(
        h_values, soft_pct, color="teal", lw=1.5, ls="--", alpha=0.8,
        label="Soft coverage (optimization objective)",
    )
    ax1.fill_between(h_values, cov_pct, alpha=0.12, color="steelblue")

    ax1.axvline(h_star, color="red", lw=2, ls="--", label=f"h* = {h_star:.0f} m")
    ax1.scatter([h_star], [cov_star * 100], color="red", s=100, zorder=5)
    ax1.annotate(
        f"  h* = {h_star:.0f} m\n  Coverage = {cov_star * 100:.1f}%",
        xy=(h_star, cov_star * 100),
        xytext=(h_star + 40, max(cov_star * 100 - 12, 5)),
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="red", alpha=0.9),
    )

    ax1.set_ylabel("Coverage (%)")
    ax1.set_title(
        f"Population-Weighted UAV-BS Altitude Optimization (Coastal Bangladesh)\n"
        f"3GPP TR 36.777 | fc = {params.fc_sub6 / 1e9:.1f} GHz | "
        f"P_tx = {params.P_tx_dBm} dBm | BW = {params.BW_MHz} MHz"
    )
    ax1.legend(fontsize=9)
    ax1.grid(True)
    ax1.set_ylim(0, 105)

    dh = h_values[1] - h_values[0]
    d_soft = np.gradient(soft_pct, dh)

    ax2.plot(
        h_values, d_soft, color="darkorange", lw=2,
        label="d(soft coverage)/dh",
    )
    ax2.axhline(0, color="black", lw=1, alpha=0.5)
    ax2.axvline(h_star, color="red", lw=2, ls="--")
    idx_near = np.argmin(np.abs(d_soft))
    ax2.scatter([h_star], [d_soft[idx_near]], color="red", s=80, zorder=5,
                label=f"h* = {h_star:.0f} m")
    ax2.set_xlabel("UAV Altitude h (m)")
    ax2.set_ylabel("d(soft cov)/dh (%/m)")
    ax2.legend(fontsize=10)
    ax2.grid(True)

    plt.tight_layout()
    path = OUTPUT_DIR / "fig2_coverage_vs_altitude.png"
    plt.savefig(path)
    plt.close()
    print(f"    Saved: {path}")
    return path


def plot_fig3_frequency_comparison(
    h_values,
    coverages_sub6, h_star_sub6, cov_star_sub6,
    coverages_mm, h_star_mm, cov_star_mm,
    params_sub6, params_mm,
):
    print("  Generating Fig 3: Sub-6 GHz vs mmWave Comparison...")

    fig, ax = plt.subplots(figsize=(11, 6))

    cov_sub6_pct = coverages_sub6 * 100
    cov_mm_pct = coverages_mm * 100

    ax.plot(
        h_values, cov_sub6_pct, color="steelblue", lw=2.5,
        label=f"Sub-6 GHz ({params_sub6.fc_sub6 / 1e9:.1f} GHz)",
    )
    ax.fill_between(h_values, cov_sub6_pct, alpha=0.1, color="steelblue")

    ax.plot(
        h_values, cov_mm_pct, color="darkorange", lw=2.5,
        label=f"mmWave ({params_mm.fc_mmwave / 1e9:.0f} GHz)",
    )
    ax.fill_between(h_values, cov_mm_pct, alpha=0.1, color="darkorange")

    ax.axvline(h_star_sub6, color="steelblue", lw=2, ls="--", alpha=0.8)
    ax.axvline(h_star_mm, color="darkorange", lw=2, ls="--", alpha=0.8)

    ax.scatter([h_star_sub6], [cov_star_sub6 * 100], color="steelblue", s=120, zorder=5)
    ax.scatter([h_star_mm], [cov_star_mm * 100], color="darkorange", s=120, zorder=5)

    ax.annotate(
        f"h*_sub6 = {h_star_sub6:.0f} m",
        xy=(h_star_sub6, cov_star_sub6 * 100),
        xytext=(h_star_sub6 - 90, min(cov_star_sub6 * 100 + 10, 95)),
        fontsize=10, color="steelblue",
        arrowprops=dict(arrowstyle="->", color="steelblue", lw=1.5),
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="steelblue", alpha=0.85),
    )
    ax.annotate(
        f"h*_mmWave = {h_star_mm:.0f} m",
        xy=(h_star_mm, cov_star_mm * 100),
        xytext=(h_star_mm + 35, min(cov_star_mm * 100 + 10, 95)),
        fontsize=10, color="darkorange",
        arrowprops=dict(arrowstyle="->", color="darkorange", lw=1.5),
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="darkorange", alpha=0.85),
    )

    delta_h = abs(h_star_mm - h_star_sub6)
    y_arrow = max(12, min(cov_star_mm, cov_star_sub6) * 100 * 0.2)
    ax.annotate(
        "", xy=(h_star_mm, y_arrow), xytext=(h_star_sub6, y_arrow),
        arrowprops=dict(arrowstyle="<->", color="gray", lw=2),
    )
    ax.text(
        (h_star_sub6 + h_star_mm) / 2, y_arrow + 3,
        f"Delta h* = {delta_h:.0f} m",
        ha="center", fontsize=10, color="gray",
    )

    ax.set_xlabel("UAV Altitude h (m)")
    ax.set_ylabel("Population-Weighted Coverage (%)")
    ax.set_title(
        "Sub-6 GHz vs mmWave Optimal Altitude Comparison\n"
        "Cox's Bazar Coastal Terrain | 3GPP TR 36.777 Aerial UMa"
    )
    ax.legend(fontsize=11)
    ax.grid(True)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    path = OUTPUT_DIR / "fig3_frequency_comparison.png"
    plt.savefig(path)
    plt.close()
    print(f"    Saved: {path}")
    return path


def plot_bonus_heatmaps(coverage_metric, h_star, params):
    print("  Generating Bonus: Population + SINR heatmaps...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ext = params.grid_extent / 1000
    pop_src = getattr(coverage_metric, "pop_source", "unknown")

    im1 = ax1.imshow(
        coverage_metric.W,
        extent=[-ext, ext, -ext, ext],
        origin="lower",
        cmap="YlOrRd",
        aspect="equal",
    )
    plt.colorbar(im1, ax=ax1, label="Relative Population Density")
    ax1.set_title(f"Population Weights ({pop_src})")
    ax1.set_xlabel("East-West (km)")
    ax1.set_ylabel("North-South (km)")

    X_km = coverage_metric.X / 1000
    Y_km = coverage_metric.Y / 1000
    ax1.contour(X_km, Y_km, coverage_metric.W, levels=5, colors="black", alpha=0.4, linewidths=0.8)

    sinr_map = coverage_metric.sinr_surface_at_altitude(h_star)
    th = params.SINR_threshold_dB

    im2 = ax2.imshow(
        sinr_map,
        extent=[-ext, ext, -ext, ext],
        origin="lower",
        cmap="RdYlGn",
        aspect="equal",
        vmin=th - 15,
        vmax=th + 25,
    )
    plt.colorbar(im2, ax=ax2, label="SINR (dB)")

    ax2.contour(
        X_km, Y_km, sinr_map, levels=[th], colors="black", linewidths=2, linestyles="--"
    )
    ax2.set_title(
        f"SINR at h* = {h_star:.0f} m (dashed = {th} dB threshold)"
    )
    ax2.set_xlabel("East-West (km)")
    ax2.set_ylabel("North-South (km)")
    ax2.plot(0, 0, "b^", markersize=15, label="UAV footprint")
    ax2.legend(fontsize=10)

    plt.tight_layout()
    path = OUTPUT_DIR / "bonus_population_sinr_maps.png"
    plt.savefig(path)
    plt.close()
    print(f"    Saved: {path}")
    return path


def plot_fig4_sensitivity(results: list, output_path: str | None = None):
    """Heatmap of h* over P_tx and control SINR threshold."""
    print("  Generating Fig 4: Sensitivity h*...")

    p_tx_vals = sorted({r["P_tx_dBm"] for r in results})
    th_vals = sorted({r["SINR_threshold_dB"] for r in results})

    grid = np.full((len(th_vals), len(p_tx_vals)), np.nan)
    for r in results:
        i = th_vals.index(r["SINR_threshold_dB"])
        j = p_tx_vals.index(r["P_tx_dBm"])
        grid[i, j] = r["h_star"]

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(grid, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xticks(range(len(p_tx_vals)))
    ax.set_xticklabels([str(v) for v in p_tx_vals])
    ax.set_yticks(range(len(th_vals)))
    ax.set_yticklabels([str(v) for v in th_vals])
    ax.set_xlabel("P_tx (dBm)")
    ax.set_ylabel("SINR threshold γ_th (dB, control channel)")
    ax.set_title("Sub-6 GHz h* vs P_tx and control SINR threshold")

    for i in range(len(th_vals)):
        for j in range(len(p_tx_vals)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.0f}", ha="center", va="center", color="white", fontsize=10)

    plt.colorbar(im, ax=ax, label="h* (m)")
    plt.tight_layout()

    path = Path(output_path) if output_path else OUTPUT_DIR / "fig4_sensitivity_hstar.png"
    plt.savefig(path)
    plt.close()
    print(f"    Saved: {path}")
    return path


def plot_fig5_data_sensitivity(results: list, output_path: str | None = None):
    """Bar chart of h* vs data-channel SINR threshold at fixed P_tx."""
    print("  Generating Fig 5: Data-threshold sensitivity h*...")

    th_vals = sorted({r["SINR_threshold_data_dB"] for r in results})
    h_stars = [
        next(r["h_star"] for r in results if r["SINR_threshold_data_dB"] == th)
        for th in th_vals
    ]
    data_cov = [
        next(r["coverage_data_pct"] for r in results if r["SINR_threshold_data_dB"] == th)
        for th in th_vals
    ]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    x = np.arange(len(th_vals))
    bars = ax1.bar(x, h_stars, color="mediumpurple", alpha=0.85, label="h* (m)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{v}" for v in th_vals])
    ax1.set_xlabel("Data SINR threshold γ_th,data (dB)")
    ax1.set_ylabel("Optimal altitude h* (m)")
    ax1.set_title("Sub-6 GHz h* vs data-channel threshold (P_tx = 27 dBm)")

    for bar, h in zip(bars, h_stars):
        ax1.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
            f"{h:.0f}", ha="center", va="bottom", fontsize=10,
        )

    ax2 = ax1.twinx()
    ax2.plot(x, data_cov, "o-", color="darkorange", lw=2, label="Data cov. at h* (%)")
    ax2.set_ylabel("Data coverage at h* (%)")
    ax2.set_ylim(0, 105)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)
    ax1.grid(True, axis="y", alpha=0.35)

    plt.tight_layout()
    path = Path(output_path) if output_path else OUTPUT_DIR / "fig5_sensitivity_data_hstar.png"
    plt.savefig(path)
    plt.close()
    print(f"    Saved: {path}")
    return path
