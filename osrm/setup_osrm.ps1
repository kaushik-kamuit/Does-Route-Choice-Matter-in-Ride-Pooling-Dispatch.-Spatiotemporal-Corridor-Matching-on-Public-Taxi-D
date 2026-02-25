# OSRM MLD Setup Script for Windows (PowerShell)
# Downloads NY metro OpenStreetMap data, builds MLD graph, launches server.
#
# Prerequisites: Docker Desktop running.
# Usage: .\osrm\setup_osrm.ps1

$ErrorActionPreference = "Stop"
$OSRM_DIR = "$PSScriptRoot"
$DATA_DIR = "$OSRM_DIR\data"
$OSM_FILE = "new-york-latest.osm.pbf"
$OSM_URL  = "https://download.geofabrik.de/north-america/us/new-york-latest.osm.pbf"

Write-Host "=== OSRM MLD Setup ===" -ForegroundColor Cyan

# --- Step 1: Create data directory ---
if (-not (Test-Path $DATA_DIR)) {
    New-Item -ItemType Directory -Path $DATA_DIR | Out-Null
    Write-Host "[1/5] Created $DATA_DIR"
} else {
    Write-Host "[1/5] Data directory exists"
}

# --- Step 2: Download OSM data ---
$OSM_PATH = "$DATA_DIR\$OSM_FILE"
if (-not (Test-Path $OSM_PATH)) {
    Write-Host "[2/5] Downloading $OSM_FILE (~500 MB, may take a few minutes)..."
    Invoke-WebRequest -Uri $OSM_URL -OutFile $OSM_PATH
    Write-Host "      Downloaded: $((Get-Item $OSM_PATH).Length / 1MB) MB"
} else {
    Write-Host "[2/5] OSM file already exists: $((Get-Item $OSM_PATH).Length / 1MB) MB"
}

# --- Step 3: OSRM Extract ---
$OSRM_FILE = "$DATA_DIR\new-york-latest.osrm"
if (-not (Test-Path $OSRM_FILE)) {
    Write-Host "[3/5] Running osrm-extract (this takes a few minutes)..."
    docker run --rm -v "${DATA_DIR}:/data" ghcr.io/project-osrm/osrm-backend osrm-extract -p /opt/car.lua /data/$OSM_FILE
    Write-Host "      Extract complete."
} else {
    Write-Host "[3/5] Extract already done, skipping."
}

# --- Step 4: OSRM Partition + Customize (MLD) ---
$PARTITION_FILE = "$DATA_DIR\new-york-latest.osrm.partition"
if (-not (Test-Path $PARTITION_FILE)) {
    Write-Host "[4/5] Running osrm-partition + osrm-customize (MLD build)..."
    docker run --rm -v "${DATA_DIR}:/data" ghcr.io/project-osrm/osrm-backend osrm-partition /data/new-york-latest.osrm
    docker run --rm -v "${DATA_DIR}:/data" ghcr.io/project-osrm/osrm-backend osrm-customize /data/new-york-latest.osrm
    Write-Host "      MLD build complete."
} else {
    Write-Host "[4/5] MLD partition already built, skipping."
}

# --- Step 5: Launch server ---
Write-Host "[5/5] Launching OSRM server on port 5000 (MLD algorithm)..."
Write-Host "      Stop with: docker stop osrm-mld"

$existing = docker ps -q --filter "name=osrm-mld"
if ($existing) {
    Write-Host "      Container 'osrm-mld' already running."
} else {
    $stopped = docker ps -aq --filter "name=osrm-mld"
    if ($stopped) {
        docker rm osrm-mld | Out-Null
    }
    docker run -d --name osrm-mld -p 5000:5000 -v "${DATA_DIR}:/data" ghcr.io/project-osrm/osrm-backend osrm-routed --algorithm mld /data/new-york-latest.osrm
}

Write-Host ""
Write-Host "=== OSRM MLD server is running ===" -ForegroundColor Green
Write-Host "    URL: http://localhost:5000"
Write-Host "    Test: curl http://localhost:5000/route/v1/driving/-73.985,40.758;-73.944,40.678?alternatives=3"
Write-Host ""
Write-Host "Set OSRM_BASE_URL=http://localhost:5000 in your .env file."
