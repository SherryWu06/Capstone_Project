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

```bash
python scripts/run_baseline.py
```

Loads data, extracts movement features, maps weeks to season labels, trains a Random Forest with time-series CV, and prints accuracy.

### Quick test

```python
from pathlib import Path
from src.raster_processing import load_weekly_stack
from src.feature_extraction import compute_global_features

stack, meta = load_weekly_stack(Path("data/raw"), species="yebsap-example", resolution="27km")
features = compute_global_features(stack)
```
