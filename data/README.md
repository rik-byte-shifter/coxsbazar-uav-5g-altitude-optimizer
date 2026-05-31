# Using WorldPop with this project

The simulator reads a GeoTIFF from this folder. If a valid file is found **and** `rasterio` is installed, all coverage metrics use **real** population weights instead of the synthetic Cox's Bazar model.

## Quick checklist

1. `pip install rasterio`
2. Download / clip WorldPop for Cox's Bazar (~10 km × 10 km)
3. Save as `data/worldpop_coxsbazar.tif` (WGS84 recommended)
4. Run `python check_worldpop.py` then `python main.py`
5. Confirm console shows: `Population weights: worldpop:worldpop_coxsbazar.tif`

---

## Step 1 — Install dependency

```powershell
cd "c:\Users\rikra\Downloads\project uav"
pip install rasterio
```

---

## Step 2 — Download WorldPop data

### Option A — WorldPop Hub (recommended)

1. Open [https://www.worldpop.org/](https://www.worldpop.org/) → **Data** → **Population Counts**.
2. Choose **Bangladesh**, year **2020** (or latest), resolution **100 m**.
3. Download the national GeoTIFF (`.tif`).

### Option B — Direct CONSTRAINED product

Search for: *WorldPop Bangladesh population count 100m 2020* — use the UN-adjusted **Population Count** raster, not density, unless you prefer it (both work as relative weights).

---

## Step 3 — Clip to Cox's Bazar AOI

Your simulation grid is **±5000 m** around the map centre in `SystemParams`:

- **Latitude centre:** `21.43°N`
- **Longitude centre:** `91.98°E`

Approximate clip box (10 km × 10 km):

| | Value |
|---|--------|
| West | 91.93°E |
| East | 92.03°E |
| South | 21.38°N |
| North | 21.48°N |

### Using GDAL (`gdalwarp`) — command line

```powershell
gdalwarp -te 91.93 21.38 92.03 21.48 -t_srs EPSG:4326 bangladesh_worldpop.tif data/worldpop_coxsbazar.tif
```

Replace `bangladesh_worldpop.tif` with your downloaded file path.

### Using QGIS

1. Load the Bangladesh WorldPop layer.
2. **Raster → Extraction → Clip raster by extent**.
3. Draw a box around Cox's Bazar (coordinates above).
4. Set output CRS to **EPSG:4326 (WGS 84)**.
5. Export as `worldpop_coxsbazar.tif` into this `data/` folder.

---

## Step 4 — Accepted file names

Any of these work (first match wins):

- `worldpop_coxsbazar.tif` ← **preferred**
- `worldpop_coxsbazar.tiff`
- `worldpop_cox_bazar.tif`
- Any `data/worldpop*.tif` / `.tiff`

---

## Step 5 — Verify before full run

From the project folder:

```powershell
python check_worldpop.py
```

You should see population min/max, grid shape, and `Source: worldpop:...`.

---

## Step 6 — Run the project

```powershell
python main.py
```

Look for:

```
Population weights: worldpop:worldpop_coxsbazar.tif
```

The bonus figure subtitle and `results_summary.txt` will also show `Population: worldpop:...`.

---

## How it works in code

- `population_data.load_population_weights()` converts each grid point `(x, y)` in **metres** (relative to UAV) to lon/lat using `lat_center` / `lon_center`.
- Values are sampled from the raster with `rasterio`.
- Weights are normalized to **[0, 1]** for the coverage sum.

To move the study area, edit in `uav_optimizer.py`:

```python
lat_center = 21.43
lon_center = 91.98
```

and re-clip your raster to match.

---

## Troubleshooting

| Problem | Fix |
|--------|-----|
| Still says `synthetic` | File not in `data/` or wrong name; run `python check_worldpop.py` |
| `rasterio` missing | `pip install rasterio` |
| All zeros / fallback warning | Clip box wrong CRS or outside Bangladesh tile; check bounds in QGIS |
| Slow first run | Normal; raster sampling is once per simulation |
| Huge national .tif without clip | Clip first — full Bangladesh file is slow and may mis-align |

---

## Optional: match grid extent

Default `grid_extent = 5000` m (±5 km). If your clip is smaller than 10 km, weights outside the raster may be zero — clip at least **10 km × 10 km** or reduce `grid_extent` in `SystemParams`.
