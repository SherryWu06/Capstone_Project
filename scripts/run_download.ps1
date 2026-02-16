# Run the eBird download script - finds R automatically
$rPaths = @(
    "C:\Program Files\R\*\bin\Rscript.exe",
    "$env:LOCALAPPDATA\Programs\R\*\bin\Rscript.exe"
)
$rscript = $null
foreach ($pattern in $rPaths) {
    $found = Get-Item $pattern -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($found) { $rscript = $found.FullName; break }
}
if (-not $rscript) {
    $rscript = (Get-Command Rscript -ErrorAction SilentlyContinue).Source
}
if (-not $rscript) {
    Write-Host "R not found. Install from https://cran.r-project.org and add to PATH."
    exit 1
}
$projectRoot = Split-Path $PSScriptRoot -Parent
Set-Location $projectRoot

# Install ebirdst if not present
& $rscript -e "if (!requireNamespace('ebirdst', quietly = TRUE)) { install.packages('ebirdst', repos='https://cloud.r-project.org') }"

# Run download
& $rscript "scripts/download_data.R"
