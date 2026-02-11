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
