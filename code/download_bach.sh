#!/usr/bin/env bash
# Download the BACH (ICIAR2018) training set from Zenodo for the Stage A smoke test.
# Training zip only (10.4GB) -- the 3GB test zip isn't needed for a smoke test.
# Run on the VM, not locally (it's big and this is meant to feed the GPU pipeline there).
# Usage: ./download_bach.sh [output_dir]   (default: data/bach)
set -euo pipefail

URL="https://zenodo.org/records/3632035/files/ICIAR2018_BACH_Challenge.zip?download=1"
MD5="8ae1801334aa943c44627c1eef3631b2"
OUT_DIR="${1:-data/bach}"
ZIP_PATH="$OUT_DIR/ICIAR2018_BACH_Challenge.zip"

mkdir -p "$OUT_DIR"
echo "Downloading BACH training set (10.4GB) to $ZIP_PATH"

# --speed-limit/--speed-time aborts a stalled transfer (no server-side error, just
# a dead connection) after 30s under 2KB/s -- curl has no stall detection otherwise,
# so a hung connection sits forever even after the network recovers.
#
# Deliberately NOT using curl's own --retry here: --retry recomputes the -C -
# resume offset only ONCE at process start, not before each internal retry. If
# --speed-limit aborts mid-transfer and --retry silently retries within the same
# process, it reuses that stale offset and overwrites/truncates everything written
# since (verified -- lost ~3GB of real progress to exactly this). Every retry must
# be a fresh process so -C - re-reads the current file size; that's this loop.
for attempt in 1 2 3 4 5 6 7 8 9 10; do
  if curl -L -C - --speed-time 30 --speed-limit 2048 -o "$ZIP_PATH" "$URL"; then
    break
  fi
  echo "curl attempt $attempt stalled/failed, retrying as a fresh process..."
  sleep 5
done

echo "Verifying checksum..."
echo "$MD5  $ZIP_PATH" | md5sum -c -

echo "Extracting..."
unzip -q -o "$ZIP_PATH" -d "$OUT_DIR"

echo "Done. Data in $OUT_DIR"
