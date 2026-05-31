"""
Live visualizations for the UAV 5G altitude optimization project.

Primary deliverable:
  outputs/live_full_project.html — full simulation playback (all phases)

Also:
  outputs/live_3d_optimizer.html — Sub-6 h-d-SINR optimizer only
  outputs/live_ground_sinr.html    — ground-map sweep
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


@dataclass
class _PhaseFrame:
    name: str
    phase: str
    data: list
    layout_patch: dict


def _camera(azim_deg: float, elev: float = 28):
    rad = np.deg2rad(azim_deg)
    er = np.deg2rad(elev)
    return dict(
        eye=dict(
            x=1.7 * np.cos(rad) * np.cos(er),
            y=1.7 * np.sin(rad) * np.cos(er),
            z=1.1 * np.sin(er),
        )
    )


def _banner(phase: str, detail: str, step: int, total: int) -> list:
    return [
        dict(
            text=(
                f"<b>UAV 5G Full Project Simulation</b>   "
                f"[{step}/{total}] <br>"
                f"<b style='color:#1a5276'>{phase}</b><br>{detail}"
            ),
            xref="paper",
            yref="paper",
            x=0.5,
            y=1.02,
            xanchor="center",
            yanchor="bottom",
            showarrow=False,
            font=dict(size=13),
        )
    ]


def _build_hd_surface(sinr_calc, params, n_h: int = 42, n_d: int = 42):
    h_arr = np.linspace(params.h_min, params.h_max, n_h)
    d_arr = np.linspace(1, params.grid_extent, n_d)
    H, D = np.meshgrid(h_arr, d_arr, indexing="ij")
    Z = np.zeros_like(H, dtype=float)
    for i, h in enumerate(h_arr):
        Z[i, :] = sinr_calc.sinr_grid(h, D[i, :])
    return H, D, Z


def _sweep_indices(h_values, h_star, frame_step: int) -> list[int]:
    idx_star = int(np.argmin(np.abs(h_values - h_star)))
    idxs = list(range(0, len(h_values), frame_step))
    if idx_star not in idxs:
        idxs.append(idx_star)
    return sorted(set(idxs))


def _coverage_traces(h_values, coverages, soft, h_star, row, col, color, name):
    """Static curve + moving marker traces for subplot (row, col)."""
    import plotly.graph_objects as go

    cov_pct = coverages * 100
    soft_pct = soft * 100
    curve = go.Scatter(
        x=h_values,
        y=cov_pct,
        mode="lines",
        line=dict(color=color, width=2),
        name=f"{name} binary %",
        xaxis=f"x{col if col > 1 else ''}",
        yaxis=f"y{col if col > 1 else ''}",
    )
    soft_curve = go.Scatter(
        x=h_values,
        y=soft_pct,
        mode="lines",
        line=dict(color=color, width=1, dash="dot"),
        opacity=0.5,
        name=f"{name} soft %",
        showlegend=False,
        xaxis=f"x{col if col > 1 else ''}",
        yaxis=f"y{col if col > 1 else ''}",
    )
    h_line = go.Scatter(
        x=[h_star, h_star],
        y=[0, 100],
        mode="lines",
        line=dict(color=color, width=1, dash="dash"),
        opacity=0.4,
        showlegend=False,
        xaxis=f"x{col if col > 1 else ''}",
        yaxis=f"y{col if col > 1 else ''}",
    )
    marker = go.Scatter(
        x=[float(h_values[0])],
        y=[float(cov_pct[0])],
        mode="markers",
        marker=dict(size=12, color=color, symbol="circle", line=dict(width=2, color="white")),
        name=f"{name} UAV h",
        xaxis=f"x{col if col > 1 else ''}",
        yaxis=f"y{col if col > 1 else ''}",
    )
    return curve, soft_curve, h_line, marker


def generate_full_project_simulation(
    res_sub6: dict,
    res_mm: dict,
    frame_step: int = 6,
) -> str:
    """
    Single HTML that plays the entire project simulation:
      intro -> population -> Sub-6 3D sweep -> Sub-6 ground @ h* ->
      mmWave 3D sweep -> frequency comparison -> finale
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("  plotly not installed; skip full project visualizer.")
        return ""

    # --- unpack ---
    cov6 = res_sub6["cov_metric"]
    covm = res_mm["cov_metric"]
    p6 = cov6.p
    pm = covm.p
    sinr6 = res_sub6["sinr_calc"]
    sinrm = res_mm["sinr_calc"]
    h6 = res_sub6["h_values"]
    h6_star = res_sub6["h_star"]
    c6 = res_sub6["coverages"]
    s6 = res_sub6["soft_coverages"]
    hm = res_mm["h_values"]
    h_mm_star = res_mm["h_star"]
    cm = res_mm["coverages"]
    sm = res_mm["soft_coverages"]
    pop_src = cov6.pop_source
    th = p6.SINR_threshold_dB

    x_km = cov6.X / 1000
    y_km = cov6.Y / 1000
    W = cov6.W

    H6, D6, Z6 = _build_hd_surface(sinr6, p6)
    Hm, Dm, Zm = _build_hd_surface(sinrm, pm)
    d_ridge = np.linspace(1, p6.grid_extent, 100)
    x_star6 = [h6_star] * len(d_ridge)
    y_star6 = (d_ridge / 1000).tolist()
    z_star6 = sinr6.sinr_grid(h6_star, d_ridge).tolist()
    x_starm = [h_mm_star] * len(d_ridge)
    z_starm = sinrm.sinr_grid(h_mm_star, d_ridge).tolist()

    # Base 2D traces (indices 4-11 after 4x 3D traces - actually 3D uses scene only)
    # Layout: row1 scene colspan 2; row2 two xy plots
    fig = make_subplots(
        rows=2,
        cols=2,
        row_heights=[0.68, 0.32],
        column_widths=[0.5, 0.5],
        specs=[
            [{"type": "scene", "colspan": 2}, None],
            [{"type": "xy"}, {"type": "xy"}],
        ],
        subplot_titles=(
            "",
            "",
            f"Sub-6 GHz ({p6.fc_sub6 / 1e9:.1f} GHz) coverage vs altitude",
            f"mmWave ({pm.fc_mmwave / 1e9:.0f} GHz) coverage vs altitude",
        ),
        vertical_spacing=0.08,
    )

    # 3D trace builders
    def _pop_surface():
        return go.Surface(
            x=x_km,
            y=y_km,
            z=W,
            colorscale="YlOrRd",
            colorbar=dict(title="Pop.", len=0.45, y=0.82),
            opacity=0.95,
        )

    def _hd_surface(H, D, Z, colorscale="Viridis", cbar_title="SINR (dB)"):
        return go.Surface(
            x=H,
            y=D / 1000,
            z=Z,
            colorscale=colorscale,
            colorbar=dict(title=cbar_title, len=0.45, y=0.82),
            opacity=0.9,
        )

    def _ground_surface(h, cov_metric, sinr_calc):
        z = cov_metric.sinr_surface_at_altitude(h)
        return go.Surface(
            x=x_km,
            y=y_km,
            z=z,
            colorscale="RdYlGn",
            colorbar=dict(title="SINR (dB)", len=0.45, y=0.82),
            cmin=th - 15,
            cmax=th + 25,
        )

    def _ridge(h, z_line, color="red", width=6):
        return go.Scatter3d(
            x=[h] * len(d_ridge),
            y=(d_ridge / 1000).tolist(),
            z=z_line.tolist() if hasattr(z_line, "tolist") else list(z_line),
            mode="lines",
            line=dict(color=color, width=width),
            showlegend=False,
        )

    def _uav_marker(h, z):
        return go.Scatter3d(
            x=[h],
            y=[0],
            z=[z],
            mode="markers",
            marker=dict(size=9, color="red", symbol="diamond"),
            showlegend=False,
        )

    def _frame_data_3d(main, ridge=None, uav=None):
        data = [main]
        if ridge is not None:
            data.append(ridge)
        else:
            data.append(
                go.Scatter3d(x=[None], y=[None], z=[None], mode="markers", opacity=0)
            )
        if uav is not None:
            data.append(uav)
        else:
            data.append(
                go.Scatter3d(x=[None], y=[None], z=[None], mode="markers", opacity=0)
            )
        return data

    # Trace order: [surf3d, ridge3d, uav3d, sub6 x4, mm x4] = 11 traces
    def _make_frame(
        name: str,
        phase: str,
        detail: str,
        surf,
        ridge,
        uav,
        h6_dot,
        c6_dot,
        hm_dot,
        cm_dot,
        az: float = -55,
        elev: float = 28,
    ):
        data = [
            surf,
            ridge if ridge else go.Scatter3d(x=[None], y=[None], z=[None], opacity=0),
            uav if uav else go.Scatter3d(x=[None], y=[None], z=[None], opacity=0),
        ]
        sub6_tr = _coverage_traces(h6, c6, s6, h6_star, 2, 1, "steelblue", "Sub-6")
        mm_tr = _coverage_traces(hm, cm, sm, h_mm_star, 2, 2, "darkorange", "mmWave")
        data.extend(sub6_tr[:3])
        data.append(
            go.Scatter(
                x=[h6_dot], y=[c6_dot], mode="markers",
                marker=dict(size=12, color="steelblue", symbol="circle",
                            line=dict(width=2, color="white")),
            )
        )
        data.extend(mm_tr[:3])
        data.append(
            go.Scatter(
                x=[hm_dot], y=[cm_dot], mode="markers",
                marker=dict(size=12, color="darkorange", symbol="circle",
                            line=dict(width=2, color="white")),
            )
        )

        return go.Frame(
            name=name,
            data=data,
            layout=go.Layout(scene=dict(camera=_camera(az, elev))),
        )

    # Build all phases
    raw_frames = []

    # 1 — Intro
    z_intro = np.zeros_like(W)
    for az in range(-60, -20, 15):
        raw_frames.append(
            _make_frame(
                f"intro_{az}",
                "Phase 1: Project start",
                f"Cox's Bazar coastal UAV 5G | 3GPP TR 36.777 | Population: {pop_src}",
                _pop_surface(),
                None,
                _uav_marker(0, 0),
                float(h6[0]), float(c6[0] * 100), float(hm[0]), float(cm[0] * 100),
                az=az,
            )
        )

    # 2 — Population 3D
    for az in range(-20, 340, 25):
        raw_frames.append(
            _make_frame(
                f"pop_{az}",
                "Phase 2: Population-weighted demand map",
                "Coastal strip, town, and camp clusters drive coverage metric",
                _pop_surface(),
                None,
                _uav_marker(0, float(W.max())),
                float(h6[0]), float(c6[0] * 100), float(hm[0]), float(cm[0] * 100),
                az=az,
                elev=35,
            )
        )

    # 3 — Sub-6 full surface
    raw_frames.append(
        _make_frame(
            "sub6_surface",
            "Phase 3: Sub-6 GHz link model",
            f"3D SINR surface PL(h,d) | P_tx={p6.P_tx_dBm} dBm, threshold={th} dB",
            _hd_surface(H6, D6, Z6),
            _ridge(h6_star, z_star6, color="gold", width=3),
            None,
            float(h6[0]), float(c6[0] * 100), float(hm[0]), float(cm[0] * 100),
            az=-45,
        )
    )

    # 4 — Sub-6 altitude sweep (optimizer)
    idx6 = _sweep_indices(h6, h6_star, frame_step)
    for k, i in enumerate(idx6):
        h = float(h6[i])
        z_line = sinr6.sinr_grid(h, d_ridge)
        at_star = i == int(np.argmin(np.abs(h6 - h6_star)))
        raw_frames.append(
            _make_frame(
                f"sub6_sweep_{h:.0f}",
                "Phase 4: Sub-6 altitude optimization",
                (
                    f"Searching h = {h:.0f} m | cov = {c6[i]*100:.1f}%"
                    + (" -> converged h*" if at_star else "")
                ),
                _hd_surface(H6, D6, Z6),
                _ridge(h, z_line, color="red", width=7),
                _uav_marker(h, float(sinr6.sinr_dB(h, 1.0))),
                h, float(c6[i] * 100), float(hm[0]), float(cm[0] * 100),
                az=-50 + k * 4,
            )
        )

    # 5 — Sub-6 ground @ h*
    z_g6 = cov6.sinr_surface_at_altitude(h6_star)
    for az in range(0, 360, 30):
        raw_frames.append(
            _make_frame(
                f"sub6_ground_{az}",
                "Phase 5: Sub-6 coverage map at h*",
                f"Ground SINR at h* = {h6_star:.0f} m | binary coverage = {res_sub6['cov_star']*100:.1f}%",
                _ground_surface(h6_star, cov6, sinr6),
                None,
                _uav_marker(h6_star, float(z_g6[len(z_g6)//2, len(z_g6[0])//2])),
                h6_star, float(res_sub6["cov_star"] * 100),
                float(hm[0]), float(cm[0] * 100),
                az=az,
                elev=32,
            )
        )

    # 6 — mmWave surface
    raw_frames.append(
        _make_frame(
            "mm_surface",
            "Phase 6: mmWave link model",
            f"fc = {pm.fc_mmwave/1e9:.0f} GHz — higher path loss, lower coverage",
            _hd_surface(Hm, Dm, Zm, colorscale="Plasma"),
            _ridge(h_mm_star, z_starm, color="gold", width=3),
            None,
            h6_star, float(res_sub6["cov_star"] * 100),
            float(hm[0]), float(cm[0] * 100),
            az=-40,
        )
    )

    # 7 — mmWave sweep
    idxm = _sweep_indices(hm, h_mm_star, frame_step)
    for k, i in enumerate(idxm):
        h = float(hm[i])
        z_line = sinrm.sinr_grid(h, d_ridge)
        raw_frames.append(
            _make_frame(
                f"mm_sweep_{h:.0f}",
                "Phase 7: mmWave altitude optimization",
                f"Searching h = {h:.0f} m | cov = {cm[i]*100:.1f}%",
                _hd_surface(Hm, Dm, Zm, colorscale="Plasma"),
                _ridge(h, z_line, color="darkorange", width=7),
                _uav_marker(h, float(sinrm.sinr_dB(h, 1.0))),
                h6_star, float(res_sub6["cov_star"] * 100),
                h, float(cm[i] * 100),
                az=-48 + k * 4,
            )
        )

    # 8 — Comparison (both h* on 2D; dual ridge in 3D)
    dual_ridge = go.Scatter3d(
        x=x_star6 + [None] + x_starm,
        y=y_star6 + [None] + y_star6,
        z=z_star6 + [None] + z_starm,
        mode="lines",
        line=dict(color="purple", width=5),
        showlegend=False,
    )
    for az in range(-30, 330, 40):
        raw_frames.append(
            _make_frame(
                f"compare_{az}",
                "Phase 8: Sub-6 vs mmWave comparison",
                (
                    f"h*_Sub6 = {h6_star:.0f} m ({res_sub6['cov_star']*100:.1f}%) | "
                    f"h*_mmW = {h_mm_star:.0f} m ({res_mm['cov_star']*100:.1f}%) | "
                    f"Delta h* = {h_mm_star - h6_star:.0f} m"
                ),
                _hd_surface(H6, D6, Z6, colorscale="Viridis"),
                dual_ridge,
                None,
                h6_star, float(res_sub6["cov_star"] * 100),
                h_mm_star, float(res_mm["cov_star"] * 100),
                az=az,
            )
        )

    # 9 — Finale
    delta = h_mm_star - h6_star
    for az in range(0, 360, 20):
        raw_frames.append(
            _make_frame(
                f"finale_{az}",
                "Phase 9: Results summary",
                (
                    f"Sub-6 h*={h6_star:.0f}m cov={res_sub6['cov_star']*100:.1f}% | "
                    f"mmWave h*={h_mm_star:.0f}m cov={res_mm['cov_star']*100:.1f}% | "
                    f"Mean SINR @ h*: {res_sub6['opt']['mean_sinr_at_h_star']:.1f} / "
                    f"{res_mm['opt']['mean_sinr_at_h_star']:.1f} dB"
                ),
                _ground_surface(h6_star, cov6, sinr6),
                _ridge(h6_star, z_star6, color="gold", width=5),
                _uav_marker(h6_star, float(res_sub6["opt"]["mean_sinr_at_h_star"])),
                h6_star, float(res_sub6["cov_star"] * 100),
                h_mm_star, float(res_mm["cov_star"] * 100),
                az=az,
                elev=25,
            )
        )

    total = len(raw_frames)
    phase_map = {
        "intro": "Phase 1: Project start",
        "pop": "Phase 2: Population map",
        "sub6": "Phase 3-5: Sub-6 GHz",
        "mm": "Phase 6-7: mmWave",
        "compare": "Phase 8: Comparison",
        "finale": "Phase 9: Summary",
    }
    frames = []
    for n, fr in enumerate(raw_frames):
        phase_title = "Simulation"
        for key, title in phase_map.items():
            if fr.name.startswith(key):
                phase_title = title
                break
        frames.append(
            go.Frame(
                name=fr.name,
                data=fr.data,
                layout=go.Layout(
                    annotations=_banner(phase_title, fr.name.replace("_", " "), n + 1, total),
                    scene=dict(camera=fr.layout.scene.camera),
                ),
            )
        )

    # Initial figure from first frame
    fig = go.Figure(data=frames[0].data if frames else [], frames=frames)

    fig.update_layout(
        title=dict(
            text=(
                "<b>Live Full Project Simulation</b><br>"
                "<sup>UAV 5G altitude optimization — Cox's Bazar | Play entire pipeline</sup>"
            ),
            x=0.5,
            xanchor="center",
        ),
        height=860,
        margin=dict(t=100, b=50),
        scene=dict(
            xaxis_title="Axis 1 (h or E-W km)",
            yaxis_title="Axis 2 (d or N-S km)",
            zaxis_title="SINR / population",
            aspectmode="manual",
            aspectratio=dict(x=1.1, y=1, z=0.5),
            camera=_camera(-55),
            domain=dict(x=[0, 1], y=[0.36, 1]),
        ),
        xaxis3=dict(title="Altitude h (m)", domain=[0, 0.44], anchor="y3"),
        yaxis3=dict(title="Coverage (%)", domain=[0.22, 0.32], anchor="x3"),
        xaxis4=dict(title="Altitude h (m)", domain=[0.56, 1], anchor="y4"),
        yaxis4=dict(title="Coverage (%)", domain=[0.22, 0.32], anchor="x4"),
        updatemenus=[
            dict(
                type="buttons",
                x=0.08,
                y=0.01,
                buttons=[
                    dict(
                        label="Play full simulation",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": 160, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 60},
                            },
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            {"frame": {"duration": 0}, "mode": "immediate"},
                        ],
                    ),
                    dict(
                        label="Restart",
                        method="animate",
                        args=[
                            [frames[0].name],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    ),
                ],
            ),
        ],
        sliders=[
            dict(
                active=0,
                pad=dict(t=50),
                len=0.9,
                x=0.05,
                currentvalue=dict(prefix="Scene: "),
                steps=[
                    dict(
                        method="animate",
                        args=[
                            [fr.name],
                            {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"},
                        ],
                        label=fr.name[:18],
                    )
                    for fr in frames[:: max(1, len(frames) // 35)]
                ],
            )
        ],
    )

    path = os.path.join(OUTPUT_DIR, "live_full_project.html")
    fig.write_html(
        path,
        include_plotlyjs="cdn",
        auto_play=False,
        config={"scrollZoom": True, "displayModeBar": True},
    )
    return path


def generate_live_3d_optimizer(res_sub6: dict, frame_step: int = 5) -> str:
    """Sub-6 only: h-d-SINR optimizer animation."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return ""

    sinr_calc = res_sub6["sinr_calc"]
    params = res_sub6["cov_metric"].p
    h_values = res_sub6["h_values"]
    coverages = res_sub6["coverages"]
    soft_coverages = res_sub6["soft_coverages"]
    h_star = res_sub6["h_star"]
    th = params.SINR_threshold_dB

    H, D, Z = _build_hd_surface(sinr_calc, params)
    d_ridge = np.linspace(1, params.grid_extent, 120)
    x_star = [h_star] * len(d_ridge)
    y_star = (d_ridge / 1000).tolist()
    z_star_list = sinr_calc.sinr_grid(h_star, d_ridge).tolist()

    surface = go.Surface(x=H, y=D / 1000, z=Z, colorscale="Viridis", opacity=0.92)
    thresh_plane = go.Surface(
        x=np.meshgrid([params.h_min, params.h_max], [1, params.grid_extent / 1000])[0],
        y=np.array([[1, params.grid_extent / 1000], [1, params.grid_extent / 1000]]),
        z=np.full((2, 2), th),
        showscale=False,
        opacity=0.12,
        colorscale=[[0, "orange"], [1, "orange"]],
    )

    idx_star = int(np.argmin(np.abs(h_values - h_star)))
    sweep_idx = _sweep_indices(h_values, h_star, frame_step)

    def _ann(h, cov, soft, phase):
        return [dict(
            text=f"<b>3D Optimizer</b><br>{phase}<br>h={h:.0f}m cov={cov:.1f}% h*={h_star:.0f}m",
            xref="paper", yref="paper", x=0.02, y=0.98, showarrow=False,
            bgcolor="rgba(255,255,255,0.85)",
        )]

    frames = []
    for k, i in enumerate(sweep_idx):
        h = float(h_values[i])
        z_line = sinr_calc.sinr_grid(h, d_ridge)
        frames.append(go.Frame(
            data=[
                surface, thresh_plane,
                go.Scatter3d(x=x_star, y=y_star, z=z_star_list, mode="lines",
                             line=dict(color="gold", width=4), opacity=0.4),
                go.Scatter3d(x=[h] * len(d_ridge), y=y_star, z=z_line.tolist(),
                             mode="lines", line=dict(color="red", width=7)),
                go.Scatter3d(x=[h], y=[0], z=[float(sinr_calc.sinr_dB(h, 1))],
                             mode="markers", marker=dict(size=8, color="red")),
            ],
            name=f"h={h:.0f}",
            layout=go.Layout(
                annotations=_ann(h, coverages[i] * 100, soft_coverages[i] * 100,
                                 "Converged" if i == idx_star else "Searching"),
                scene_camera=_camera(-50 + 8 * k),
            ),
        ))

    fig = go.Figure(data=frames[0].data, frames=frames)
    fig.update_layout(
        title="Live 3D Optimizer — Sub-6 GHz",
        scene=dict(
            xaxis_title="h (m)", yaxis_title="d (km)", zaxis_title="SINR (dB)",
            camera=_camera(-50),
        ),
        updatemenus=[dict(type="buttons", buttons=[dict(
            label="Play", method="animate",
            args=[None, {"frame": {"duration": 180, "redraw": True}}],
        )])],
        height=720,
    )
    path = os.path.join(OUTPUT_DIR, "live_3d_optimizer.html")
    fig.write_html(path, include_plotlyjs="cdn")
    return path


def generate_ground_map(res_sub6: dict, frame_step: int = 8) -> str:
    try:
        import plotly.graph_objects as go
    except ImportError:
        return ""

    cov_metric = res_sub6["cov_metric"]
    h_values = res_sub6["h_values"]
    x_km = cov_metric.X / 1000
    y_km = cov_metric.Y / 1000
    frames = [
        go.Frame(
            data=[go.Surface(x=x_km, y=y_km, z=cov_metric.sinr_surface_at_altitude(h),
                             colorscale="Viridis")],
            name=f"{h:.0f}m",
        )
        for h in h_values[::frame_step]
    ]
    fig = go.Figure(data=frames[0].data, frames=frames)
    fig.update_layout(
        updatemenus=[dict(type="buttons", buttons=[dict(
            label="Play", method="animate", args=[None, {"frame": {"duration": 120}}],
        )])],
    )
    path = os.path.join(OUTPUT_DIR, "live_ground_sinr.html")
    fig.write_html(path, include_plotlyjs="cdn")
    return path


def export_static_figures_for_report(res_sub6: dict, res_mm: dict) -> list[str]:
    """Matplotlib PNG snapshots of live HTML views (for LaTeX report)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    saved = []
    cov6 = res_sub6["cov_metric"]
    sinr6 = res_sub6["sinr_calc"]
    p6 = cov6.p
    h_star = res_sub6["h_star"]

    # --- preview: live 3D optimizer (h-d SINR + ridge at h*) ---
    h_arr = np.linspace(p6.h_min, p6.h_max, 50)
    d_arr = np.linspace(1, p6.grid_extent, 50)
    H, D = np.meshgrid(h_arr, d_arr, indexing="ij")
    Z = np.zeros_like(H)
    for i, h in enumerate(h_arr):
        Z[i, :] = sinr6.sinr_grid(h, D[i, :])
    d_ridge = np.linspace(1, p6.grid_extent, 100)
    z_ridge = sinr6.sinr_grid(h_star, d_ridge)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(H, D / 1000, Z, cmap="viridis", alpha=0.9, linewidth=0)
    ax.plot([h_star] * len(d_ridge), d_ridge / 1000, z_ridge, "r-", lw=3)
    ax.set_xlabel("h (m)")
    ax.set_ylabel("d (km)")
    ax.set_zlabel("SINR (dB)")
    ax.set_title(f"Live 3D optimizer snapshot at h* = {h_star:.0f} m")
    p1 = os.path.join(OUTPUT_DIR, "preview_live_3d_optimizer.png")
    fig.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close(fig)
    saved.append(p1)

    # --- preview: live ground SINR map ---
    ext = p6.grid_extent / 1000
    sinr_map = cov6.sinr_surface_at_altitude(h_star)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(
        sinr_map,
        extent=[-ext, ext, -ext, ext],
        origin="lower",
        cmap="RdYlGn",
        vmin=p6.SINR_threshold_dB - 15,
        vmax=p6.SINR_threshold_dB + 25,
    )
    plt.colorbar(im, ax=ax, label="SINR (dB)")
    ax.set_title(f"Live ground SINR map at h* = {h_star:.0f} m")
    ax.set_xlabel("East-West (km)")
    ax.set_ylabel("North-South (km)")
    p2 = os.path.join(OUTPUT_DIR, "preview_live_ground_sinr.png")
    fig.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close(fig)
    saved.append(p2)

    # --- preview: full project (population + both coverage curves) ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].imshow(
        cov6.W,
        extent=[-ext, ext, -ext, ext],
        origin="lower",
        cmap="YlOrRd",
        aspect="equal",
    )
    axes[0].set_title("WorldPop weights")
    axes[0].set_xlabel("E-W (km)")
    axes[0].set_ylabel("N-S (km)")

    h6 = res_sub6["h_values"]
    axes[1].plot(h6, res_sub6["coverages"] * 100, "b-", lw=2)
    axes[1].axvline(h_star, color="r", ls="--")
    axes[1].set_title("Sub-6 coverage vs h")
    axes[1].set_xlabel("h (m)")
    axes[1].set_ylabel("Coverage (%)")
    axes[1].grid(True, alpha=0.3)

    hm = res_mm["h_values"]
    axes[2].plot(hm, res_mm["coverages"] * 100, color="darkorange", lw=2)
    axes[2].axvline(res_mm["h_star"], color="r", ls="--")
    axes[2].set_title("mmWave coverage vs h")
    axes[2].set_xlabel("h (m)")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Live full project simulation — summary panels", fontsize=12)
    plt.tight_layout()
    p3 = os.path.join(OUTPUT_DIR, "preview_live_full_project.png")
    fig.savefig(p3, dpi=200, bbox_inches="tight")
    plt.close(fig)
    saved.append(p3)

    return saved


def generate(res_sub6: dict, res_mm: dict | None = None, frame_step: int = 6) -> str:
    """
    Build live HTML outputs. Primary: full project simulation.
    """
    if res_mm is None:
        res_mm = res_sub6

    path_full = generate_full_project_simulation(res_sub6, res_mm, frame_step=frame_step)
    path_3d = generate_live_3d_optimizer(res_sub6, frame_step=frame_step)
    path_ground = generate_ground_map(res_sub6, frame_step=frame_step * 2)

    if path_full:
        print(f"    Live FULL project:  {path_full}")
    if path_3d:
        print(f"    Live 3D optimizer:  {path_3d}")
    if path_ground:
        print(f"    Live ground map:    {path_ground}")

    try:
        previews = export_static_figures_for_report(res_sub6, res_mm)
        for p in previews:
            print(f"    Report preview:     {p}")
    except Exception as exc:
        print(f"    Report previews skipped: {exc}")

    if path_full:
        import shutil
        for alias in ("live_sinr_optimizer.html",):
            try:
                shutil.copy2(path_full, os.path.join(OUTPUT_DIR, alias))
            except OSError:
                pass

    return path_full or path_3d or path_ground


if __name__ == "__main__":
    from main import run_pipeline

    res_sub6, res_mm, _, _, _, _ = run_pipeline()
    path = generate(res_sub6, res_mm)
    if path:
        print(f"\n  Open full simulation: {path}")
    else:
        print("\n  pip install plotly")
