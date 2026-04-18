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

**Resolution / species without editing `config.R`:**

```bash
# 9 km for specific codes (adds *_9km_*.tif next to existing 3 km in each species folder)
Rscript scripts/download_data.R --resolution 9km --species dickci swahaw yerwar
```

Or use the helper (edit `SPECIES` inside to match your `run.sh`):

```bash
bash scripts/download_ebirdst_9km.sh
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

**Phase 2 – Regional (MIL, semi-supervised):**
```bash
python scripts/run_baseline.py --regional
```
Uses Attention-based Multiple Instance Learning: bag = week (all cells), bag label = week-level. Learns which cells contribute to the label. Use `--cell-size 32` for finer grid (default: 64).

**Multiple ebirdst species (explicit codes — preferred):**
```bash
python scripts/run_baseline.py --ebirdst-species boboli dickci savspa --regional --resolution 3km --output-dir outputs/
```
Use exact species codes; data must exist under `data/raw/2023/<code>/` for the chosen resolution. For quick tests only, `--ebirdst-all` with `--max-species` / `--species-offset` still selects the first *N* sorted species.

**Species-level train/test split:**
```bash
python scripts/run_baseline.py --matt --species-split
```
Trains on 80% of species, evaluates on held-out 20% to test generalization to unseen species.

**Save outputs (optional):**
```bash
python scripts/run_baseline.py --matt --regional --output-dir outputs/
```
- **MIL:** Saves per-week attention maps and aggregate-by-season maps (`.npy`) per species
- **RF:** Saves predictions CSV (species, week_idx, actual, predicted)

### Plot attention maps

After running the MIL baseline with `--output-dir`, generate presentation maps:

```bash
python scripts/plot_attention_maps.py --attention-dir outputs/mil_inspect/test --species acafly comyel --basemap --region north_america
```

Use `--attention-dir outputs/mil_inspect/train` for train species. Add `--weekly` for per-week maps.

### Migration onset analysis

Two scripts analyse *where* and *when* movement begins, independently of the MIL classifier.  Both use a z-score threshold on week-to-week abundance change to determine the first week each spatial cell shows significant movement.  Search windows default to eBird season dates (± a buffer) when available, with fixed fallbacks otherwise.

**Static PNG maps** (`plot_migration_onset.py`):

```bash
# Per-week pixel-level movement maps
python scripts/plot_migration_onset.py --species acafly --weekly --basemap --region north_america

# Migration onset map (spring + fall, one PNG each)
python scripts/plot_migration_onset.py --species acafly norhar2 --onset --basemap --region lower_48

# Fully manual search windows
python scripts/plot_migration_onset.py --species acafly --onset \
    --onset-spring-start 8 --onset-spring-end 25 \
    --onset-fall-start 30 --onset-fall-end 48
```

Outputs: `{species}_spring_onset.png`, `{species}_fall_onset.png`, and (if `--weekly`) per-week PNGs under `movement/{species}/`.

**Interactive HTML maps** (`plot_onset_interactive.py`):

```bash
# Default: lower_48 region, 27 km resolution, cell size 16
python scripts/plot_onset_interactive.py --species acafly comyel

# Higher resolution with basemap borders
python scripts/plot_onset_interactive.py --species acafly --resolution 3km --cell-size 3 --basemap --region lower_48_plus

# Shorter chart title (presentation mode)
python scripts/plot_onset_interactive.py --species acafly --clean
```

Outputs per species: `{species}_spring_onset_interactive.html` and `{species}_fall_onset_interactive.html`.  Hovering shows the week index and calendar date label.  Colors cycle through a 7-step colorblind-safe palette anchored by month.

**Key flags (both scripts):**

| Flag | Description |
|------|-------------|
| `--resolution` | `3km`, `9km`, or `27km` (default: `27km`) |
| `--cell-size` | Grid cell size in pixels for onset aggregation (default: 16) |
| `--region` | `full`, `north_america`, `americas`, `lower_48`, `lower_48_plus` |
| `--basemap` | Overlay Natural Earth country and state/province borders |
| `--z-threshold` | Z-score threshold for onset detection (default: 1.5) |
| `--season-buffer` | Weeks of padding around eBird season dates (default: 3) |
| `--cap-weeks` | Limit display to the first N onset weeks from earliest detection |
| `--output-dir` | Output directory (default: `outputs/presentation`) |


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

## Outputs

To view generated graphs, please visit our public Google Drive folder here: https://drive.google.com/drive/folders/1-vCLHmB1t1suJsvxOqXc6i0kHHHWTtLZ?usp=drive_link