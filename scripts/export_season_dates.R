# Export ebirdst_runs season dates for species in us-breeding-species.csv
# Output: data/labels/matt_species_seasons.json (for Matt's TIF files)
# Run: Rscript scripts/export_season_dates.R

# Find project root (same logic as download_data.R)
args <- commandArgs(trailingOnly = FALSE)
file_arg <- args[grep("^--file=", args)]
if (length(file_arg) > 0) {
  script_path <- normalizePath(sub("^--file=", "", file_arg))
  project_root <- dirname(dirname(script_path))
} else {
  project_root <- getwd()
}

csv_path <- file.path(project_root, "data", "raw", "Matt", "us-breeding-species.csv")
output_path <- file.path(project_root, "data", "labels", "matt_species_seasons.json")

if (!file.exists(csv_path)) {
  stop("Species CSV not found: ", csv_path)
}

dir.create(dirname(output_path), showWarnings = FALSE, recursive = TRUE)

if (!requireNamespace("ebirdst", quietly = TRUE)) {
  stop("ebirdst required. Install with: install.packages('ebirdst')")
}
if (!requireNamespace("jsonlite", quietly = TRUE)) {
  stop("jsonlite required. Install with: install.packages('jsonlite')")
}
library(ebirdst)
library(jsonlite)

species_codes <- read.csv(csv_path)$species_code
runs <- ebirdst_runs[ebirdst_runs$species_code %in% species_codes, ]

# Standard 52-week dates (MM-DD) for eBird Status
date_names <- c("01-04", "01-11", "01-18", "01-25", "02-01", "02-08", "02-15", "02-22",
  "03-01", "03-08", "03-15", "03-22", "03-29", "04-05", "04-12", "04-19", "04-26",
  "05-03", "05-10", "05-17", "05-24", "05-31", "06-07", "06-14", "06-21", "06-28",
  "07-05", "07-12", "07-19", "07-26", "08-02", "08-09", "08-16", "08-23", "08-30",
  "09-06", "09-13", "09-20", "09-27", "10-04", "10-11", "10-18", "10-25", "11-01",
  "11-08", "11-15", "11-22", "11-29", "12-06", "12-13", "12-20", "12-27")

pred_year <- ebirdst_version()$status_version_year
year_str <- as.character(pred_year)

result <- list()
for (i in seq_len(nrow(runs))) {
  row <- runs[i, ]
  species <- row$species_code
  seasons <- list()

  if (!is.na(row$breeding_start) && !is.na(row$breeding_end)) {
    seasons <- c(seasons, list(list(
      season = "breeding",
      start_date = format(as.Date(row$breeding_start), "%Y-%m-%d"),
      end_date = format(as.Date(row$breeding_end), "%Y-%m-%d")
    )))
  }
  if (!is.na(row$nonbreeding_start) && !is.na(row$nonbreeding_end)) {
    seasons <- c(seasons, list(list(
      season = "nonbreeding",
      start_date = format(as.Date(row$nonbreeding_start), "%Y-%m-%d"),
      end_date = format(as.Date(row$nonbreeding_end), "%Y-%m-%d")
    )))
  }
  if (!is.na(row$prebreeding_migration_start) && !is.na(row$prebreeding_migration_end)) {
    seasons <- c(seasons, list(list(
      season = "prebreeding_migration",
      start_date = format(as.Date(row$prebreeding_migration_start), "%Y-%m-%d"),
      end_date = format(as.Date(row$prebreeding_migration_end), "%Y-%m-%d")
    )))
  }
  if (!is.na(row$postbreeding_migration_start) && !is.na(row$postbreeding_migration_end)) {
    seasons <- c(seasons, list(list(
      season = "postbreeding_migration",
      start_date = format(as.Date(row$postbreeding_migration_start), "%Y-%m-%d"),
      end_date = format(as.Date(row$postbreeding_migration_end), "%Y-%m-%d")
    )))
  }

  if (length(seasons) > 0) {
    result[[species]] <- list(
      species_code = species,
      year = pred_year,
      season_dates = seasons,
      DATE_NAMES = date_names
    )
  }
}

write_json(result, output_path, pretty = TRUE, auto_unbox = TRUE)
message("Wrote ", output_path, " (", length(result), " species with season data)")
message("Matched ", nrow(runs), " of ", length(species_codes), " species from CSV")
