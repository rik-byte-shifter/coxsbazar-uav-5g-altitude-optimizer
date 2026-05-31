"""
Population weighting for Cox's Bazar AOI.
WorldPop raster if present in data/, else synthetic model.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
WORLDPOP_NAMES = (
    "worldpop_coxsbazar.tif",
    "worldpop_coxsbazar.tiff",
    "worldpop_cox_bazar.tif",
    "worldpop_cox_bazar.tiff",
)
# Full-country uploads (clipped copy preferred in data/)
BANGLADESH_WORLDPOP_GLOB = (
    "bgd_pop_*_100m_*.tif",
    "bgd_pop_*_100m_*.tiff",
    "worldpop*.tif",
    "worldpop*.tiff",
)


def cox_bazar_population_density(x_grid: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
    """
    Synthetic population density for Cox's Bazar coastal terrain.
    x_grid, y_grid: 2D arrays [m] relative to UAV ground projection.
    """
    density = np.zeros_like(x_grid, dtype=float)

    coastal_weight = np.exp(-0.5 * ((x_grid + 1500) / 300) ** 2)
    density += 0.6 * coastal_weight

    town = np.exp(
        -0.5 * (((x_grid - 200) / 500) ** 2 + ((y_grid + 100) / 400) ** 2)
    )
    density += 0.9 * town

    camp1 = np.exp(
        -0.5 * (((x_grid - 800) / 300) ** 2 + ((y_grid - 700) / 250) ** 2)
    )
    camp2 = np.exp(
        -0.5 * (((x_grid - 600) / 200) ** 2 + ((y_grid - 500) / 200) ** 2)
    )
    density += 1.0 * camp1 + 0.8 * camp2

    v1 = np.exp(
        -0.5 * (((x_grid + 500) / 250) ** 2 + ((y_grid - 300) / 200) ** 2)
    )
    v2 = np.exp(
        -0.5 * (((x_grid - 300) / 200) ** 2 + ((y_grid + 600) / 250) ** 2)
    )
    density += 0.4 * v1 + 0.3 * v2
    density += 0.05

    mx = density.max()
    if mx > 0:
        density = density / mx
    return density


def _find_worldpop_path() -> Path | None:
    search_dirs = (DATA_DIR, PROJECT_ROOT)
    for directory in search_dirs:
        for name in WORLDPOP_NAMES:
            p = directory / name
            if p.is_file():
                return p
    for directory in search_dirs:
        for pattern in BANGLADESH_WORLDPOP_GLOB:
            matches = sorted(directory.glob(pattern))
            for p in matches:
                if p.is_file():
                    return p
        for p in sorted(directory.glob("worldpop*.tif*")):
            if p.is_file():
                return p
    return None


def _meters_to_lonlat(x_m: np.ndarray, y_m: np.ndarray, lat0: float, lon0: float):
    """Local equirectangular approximation for small AOI."""
    lat_rad = np.deg2rad(lat0)
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * np.cos(lat_rad)
    lon = lon0 + x_m / m_per_deg_lon
    lat = lat0 + y_m / m_per_deg_lat
    return lon, lat


def _sample_worldpop_raster(
    path: Path, lon: np.ndarray, lat: np.ndarray
) -> np.ndarray:
    import rasterio
    from rasterio.warp import transform

    with rasterio.open(path) as src:
        if src.crs and src.crs.to_string() != "EPSG:4326":
            xs, ys = transform("EPSG:4326", src.crs, lon.ravel(), lat.ravel())
            coords = list(zip(xs, ys))
        else:
            coords = list(zip(lon.ravel(), lat.ravel()))

        samples = np.array(list(src.sample(coords)), dtype=float)
        pop = samples[:, 0].reshape(lon.shape) if samples.ndim > 1 else samples.reshape(lon.shape)
        # Robust vectorized cleaning
        valid_mask = np.isfinite(pop) & (pop > 0)
        pop = np.where(valid_mask, pop, 0.0)
    return pop


def load_population_weights(
    x_grid: np.ndarray, y_grid: np.ndarray, params
) -> tuple[np.ndarray, str]:
    """
    Load population weights for the simulation grid.

    Returns
    -------
    W : ndarray
        Normalized weights in [0, 1].
    source_label : str
        'worldpop:<filename>' or 'synthetic'.
    """
    raster_path = _find_worldpop_path()
    if raster_path is None:
        w = cox_bazar_population_density(x_grid, y_grid)
        return w, "synthetic"

    try:
        import rasterio  # noqa: F401
    except ImportError:
        warnings.warn(
            "WorldPop raster found but rasterio is not installed; "
            "using synthetic population model.",
            stacklevel=2,
        )
        w = cox_bazar_population_density(x_grid, y_grid)
        return w, "synthetic"

    try:
        lon, lat = _meters_to_lonlat(
            x_grid, y_grid, params.lat_center, params.lon_center
        )
        pop = _sample_worldpop_raster(raster_path, lon, lat)
        mx = pop.max()
        if mx <= 0:
            warnings.warn(
                f"WorldPop raster {raster_path.name} has no positive values; "
                "using synthetic fallback.",
                stacklevel=2,
            )
            w = cox_bazar_population_density(x_grid, y_grid)
            return w, "synthetic"
        w = pop / mx
        return w, f"worldpop:{raster_path.name}"
    except Exception as exc:
        warnings.warn(
            f"Failed to read WorldPop raster ({exc}); using synthetic fallback.",
            stacklevel=2,
        )
        w = cox_bazar_population_density(x_grid, y_grid)
        return w, "synthetic"
