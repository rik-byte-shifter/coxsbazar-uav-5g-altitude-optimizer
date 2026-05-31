"""
Verify WorldPop raster setup before running main.py.

Usage:
    python check_worldpop.py
"""

import numpy as np

from population_data import DATA_DIR, _find_worldpop_path, load_population_weights
from uav_optimizer import SystemParams


def main():
    print("WorldPop setup check")
    print("=" * 50)
    print(f"Data folder: {DATA_DIR}")

    path = _find_worldpop_path()
    if path is None:
        print("\nNo WorldPop file found.")
        print("Place a clipped GeoTIFF in data/, e.g.:")
        print("  data/worldpop_coxsbazar.tif")
        print("\nSee data/README.md for download and clip steps.")
        return 1

    print(f"\nFound raster: {path.name}")

    try:
        import rasterio
    except ImportError:
        print("\nERROR: rasterio is not installed.")
        print("  pip install rasterio")
        return 1

    with rasterio.open(path) as src:
        print(f"  CRS:        {src.crs}")
        print(f"  Size:       {src.width} x {src.height}")
        print(f"  Bounds:     {src.bounds}")
        print(f"  Resolution: {src.res}")

    params = SystemParams()
    n = params.grid_size
    ext = params.grid_extent
    x = np.linspace(-ext, ext, n)
    y = np.linspace(-ext, ext, n)
    X, Y = np.meshgrid(x, y)

    W, source = load_population_weights(X, Y, params)
    print(f"\nSampling on {n}x{n} grid (+/-{ext} m around {params.lat_center}N, {params.lon_center}E)")
    print(f"  Source:     {source}")
    print(f"  Weight min: {W.min():.6f}")
    print(f"  Weight max: {W.max():.6f}")
    print(f"  Non-zero:   {(W > 0).sum()} / {W.size} cells")

    if source.startswith("worldpop"):
        print("\nOK — WorldPop is active. Run: python main.py")
        return 0

    print("\nWARNING — Falling back to synthetic (see messages above).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
