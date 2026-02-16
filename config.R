# eBird Status and Trends download configuration
# Edit this file with your access key and species. Your edits are gitignored.

# Access key from https://ebird.org/st/request
ACCESS_KEY <- "jqget4rqcv00"

# Species to download (common name, scientific name, or 6-letter code)
# Examples: "woothr", "Yellow-bellied Sapsucker", "Sphyrapicus varius"
# Use "yebsap-example" for a free test dataset (no key required)
SPECIES_CODES <- c("yebsap-example")

# Output directory (relative to project root)
OUTPUT_DIR <- "data/raw"

# Resolution filter: "3km", "9km", or "27km"
# Use "27km" for faster downloads and smaller files
RESOLUTION <- "9km"

# Download all products (TRUE) or just abundance (FALSE)
DOWNLOAD_ALL <- FALSE
