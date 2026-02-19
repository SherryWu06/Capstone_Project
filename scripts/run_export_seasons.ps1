# Export ebirdst_runs season dates for Matt's species to data/labels/matt_species_seasons.json
# Run: .\scripts\run_export_seasons.ps1

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
    Write-Host "R not found. Install from https://cran.r-project.org"
    exit 1
}
$projectRoot = Split-Path $PSScriptRoot -Parent
Set-Location $projectRoot
& $rscript "scripts/export_season_dates.R"
