# eBird Status and Trends Data Download Script
# Downloads weekly abundance GeoTIFF rasters for bird migration analysis.
# Requires: R, ebirdst package, config.R with access key

# Find project root (directory containing config.R)
args <- commandArgs(trailingOnly = FALSE)
file_arg <- args[grep("^--file=", args)]
if (length(file_arg) > 0) {
  script_path <- normalizePath(sub("^--file=", "", file_arg))
  project_root <- dirname(dirname(script_path))
} else {
  project_root <- getwd()
}
config_path <- file.path(project_root, "config.R")

if (!file.exists(config_path)) {
  stop("Config file not found: ", config_path, "\n",
       "Ensure config.R exists in the project root.")
}

source(config_path)

# Option: use species from Matt's CSV instead of SPECIES_CODES
if (exists("USE_MATT_SPECIES_CSV") && USE_MATT_SPECIES_CSV) {
  csv_path <- file.path(project_root, "data", "raw", "Matt", "us-breeding-species.csv")
  if (file.exists(csv_path)) {
    SPECIES_CODES <- read.csv(csv_path)$species_code
    message("Using ", length(SPECIES_CODES), " species from us-breeding-species.csv")
  }
}

# Optional CLI overrides (after config + Matt CSV):
#   Rscript scripts/download_data.R --resolution 9km
#   Rscript scripts/download_data.R --resolution 9km --species dickci swahaw yerwar
args_trail <- commandArgs(trailingOnly = TRUE)
i <- 1L
while (i <= length(args_trail)) {
  if (identical(args_trail[i], "--resolution") && i < length(args_trail)) {
    RESOLUTION <- args_trail[i + 1L]
    message("CLI override: RESOLUTION = ", RESOLUTION)
    i <- i + 2L
  } else if (identical(args_trail[i], "--species") && i < length(args_trail)) {
    i <- i + 1L
    sp <- character(0)
    while (i <= length(args_trail) && !grepl("^--", args_trail[i])) {
      sp <- c(sp, args_trail[i])
      i <- i + 1L
    }
    if (length(sp) > 0) {
      SPECIES_CODES <- sp
      message("CLI override: ", length(SPECIES_CODES), " species: ", paste(SPECIES_CODES, collapse = ", "))
    }
  } else {
    i <- i + 1L
  }
}

# Validate access key (not required for yebsap-example)
needs_key <- !all(SPECIES_CODES %in% c("yebsap-example"))
if (needs_key && (is.null(ACCESS_KEY) || ACCESS_KEY == "" || ACCESS_KEY == "your_key_here")) {
  stop("Access key required. Edit config.R with your key from https://ebird.org/st/request")
}

# Load ebirdst
if (!requireNamespace("ebirdst", quietly = TRUE)) {
  stop("ebirdst package required. Install with: install.packages(\"ebirdst\")")
}
library(ebirdst)

# Set access key
if (!needs_key) {
  message("Using example dataset (no key required)")
} else {
  set_ebirdst_access_key(ACCESS_KEY)
}

# Resolve output path
output_path <- normalizePath(file.path(project_root, OUTPUT_DIR), mustWork = FALSE)
if (!dir.exists(output_path)) {
  dir.create(output_path, recursive = TRUE)
  message("Created output directory: ", output_path)
}

# Resolution pattern for ebirdst (filters which GeoTIFFs to download)
# Note: yebsap-example only has 27km resolution
resolution_for_species <- function(species) {
  if (species == "yebsap-example") "27km" else RESOLUTION
}

# Download each species
downloaded <- character(0)
for (species in SPECIES_CODES) {
  res <- resolution_for_species(species)
  resolution_pattern <- paste0("_", res, "_")
  message("Downloading ", species, " (resolution: ", res, ")...")
  tryCatch({
    path <- ebirdst_download_status(
      species = species,
      path = output_path,
      pattern = resolution_pattern,
      download_all = DOWNLOAD_ALL,
      force = FALSE,
      show_progress = TRUE
    )
    downloaded <- c(downloaded, species)
    message("  -> ", path)
  }, error = function(e) {
    message("  FAILED: ", conditionMessage(e))
  })
}

# Summary
message("")
message("Download complete. Species: ", paste(downloaded, collapse = ", "))
if (length(downloaded) == 0) {
  message("No data downloaded. Check species codes and access key.")
} else {
  message("Data location: ", output_path)
  message("GeoTIFF files can be read in Python with rasterio.")
}
