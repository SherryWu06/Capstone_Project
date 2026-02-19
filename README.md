# Bird Migration Season Detection

ML-based detection of breeding, non-breeding, and migratory seasons from weekly GeoTIFF raster files of bird abundance.

## Data: eBird Status and Trends

Abundance rasters are downloaded from [eBird Status and Trends](https://science.ebird.org/en/status-and-trends) via the `ebirdst` R package.

### Setup

1. **Get an access key** (required for most species): Request at [https://ebird.org/st/request](https://ebird.org/st/request)
2. **Edit `config.R`**: Add your access key and species codes. Your edits are gitignored and will not be committed.
3. **Install R** and the ebirdst package:
   ```r
   install.packages("ebirdst")
   ```

### Download Data

From the project root:

```bash
Rscript scripts/download_data.R
```

Or from R:

```r
source("scripts/download_data.R")
```

### Config Options

| Variable | Description |
|----------|-------------|
| `ACCESS_KEY` | Your eBird Status and Trends key |
| `SPECIES_CODES` | Species to download (code, common name, or scientific name) |
| `OUTPUT_DIR` | Where to save GeoTIFFs (default: `data/raw`) |
| `RESOLUTION` | `3km`, `9km`, or `27km` |
| `DOWNLOAD_ALL` | `TRUE` for all products, `FALSE` for abundance only |

**Test without a key**: Use `SPECIES_CODES <- c("yebsap-example")` for a small Yellow-bellied Sapsucker dataset in Michigan.

**Use Matt's species list**: Add `USE_MATT_SPECIES_CSV <- TRUE` to config.R to download all species from `data/raw/Matt/us-breeding-species.csv` (requires access key).

## Python Analysis

After downloading data, use Python for ML and analysis:

```bash
pip install -r requirements.txt
```

### Structure

- `src/raster_processing.py` – Load GeoTIFF weekly abundance stacks
- `src/feature_extraction.py` – Movement metrics (centroid displacement, spatial variance, change magnitude)
- `notebooks/01_explore_data.ipynb` – EDA and visualization

### Run baseline (end-to-end)

**Phase 1 – Movement vs no movement (default):**
```bash
python scripts/run_baseline.py
```
Binary: migration = movement, breeding/nonbreeding = no movement.

**Phase 1 – Four-class (optional):**
```bash
python scripts/run_baseline.py --4class
```

**Phase 2 – Per-region prediction:**
```bash
python scripts/run_baseline.py --regional
```
Uses local features per grid cell for finer spatial granularity. Labels are global (weak supervision); infrastructure supports future per-location labels. Use `--cell-size 32` for finer grid (default: 64).

### Matt's data (data/raw/Matt/)

For TIF files from Matt (flat layout, no config.json):

1. **Export season labels** (run once; requires R + ebirdst):
   ```bash
   .\scripts\run_export_seasons.ps1
   ```
   Creates `data/labels/matt_species_seasons.json` from `us-breeding-species.csv`.

2. **Run baseline on Matt data:**
   ```bash
   python scripts/run_baseline.py --matt --species acafly
   ```

3. **Combine all sources** (ebirdst + Matt species with data):
   ```bash
   python scripts/run_baseline.py --source all
   ```

### Quick test

```python
from pathlib import Path
from src.raster_processing import load_weekly_stack
from src.feature_extraction import compute_global_features

stack, meta = load_weekly_stack(Path("data/raw"), species="yebsap-example", resolution="27km")
features = compute_global_features(stack)
```
