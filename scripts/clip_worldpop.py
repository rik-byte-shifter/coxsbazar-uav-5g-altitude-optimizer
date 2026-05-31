"""
Clip Bangladesh WorldPop raster to Cox's Bazar AOI for the UAV simulator.

Usage (from project root):
    python scripts/clip_worldpop.py
"""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_CANDIDATES = [
    PROJECT_ROOT / "bgd_pop_2026_CN_100m_R2025A_v1.tif",
    PROJECT_ROOT / "data" / "bgd_pop_2026_CN_100m_R2025A_v1.tif",
]
OUT_PATH = PROJECT_ROOT / "data" / "worldpop_coxsbazar.tif"

# Cox's Bazar AOI (~10 km), centred near SystemParams lat/lon
BOUNDS = (91.93, 21.38, 92.03, 21.48)  # left, bottom, right, top (WGS84)


def main():
    src_path = next((p for p in SRC_CANDIDATES if p.is_file()), None)
    if src_path is None:
        raise FileNotFoundError(
            "Bangladesh WorldPop .tif not found in project root or data/"
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(src_path) as src:
        window = from_bounds(*BOUNDS, transform=src.transform)
        data = src.read(1, window=window, boundless=True, fill_value=0)
        transform = src.window_transform(window)
        profile = src.profile.copy()
        nodata = src.nodata if src.nodata is not None else -99999
        data = np.where(data == nodata, 0, data)
        data = np.where(data < 0, 0, data)
        profile.update(
            dtype=data.dtype,
            height=data.shape[0],
            width=data.shape[1],
            transform=transform,
            compress="lzw",
            nodata=0,
        )
        with rasterio.open(OUT_PATH, "w", **profile) as dst:
            dst.write(data, 1)

    with rasterio.open(OUT_PATH) as clipped:
        print(f"Source:  {src_path}")
        print(f"Output:  {OUT_PATH}")
        print(f"Size:    {clipped.width} x {clipped.height}")
        print(f"Bounds:  {clipped.bounds}")
        print(f"CRS:     {clipped.crs}")
        arr = clipped.read(1)
        print(f"Pop min/max: {arr.min():.2f} / {arr.max():.2f}")
        print(f"Non-zero cells: {(arr > 0).sum()} / {arr.size}")


if __name__ == "__main__":
    main()
