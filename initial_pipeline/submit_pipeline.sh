#!/bin/bash
# submit_pipeline.sh
#
# Builds scan_list.txt from the scan data directory, then submits a SLURM
# array job with one task per scan.
#
# Usage (from initial_pipeline/):
#   bash submit_pipeline.sh

set -euo pipefail

SCAN_DIR="../../ScanData/10_patients_GM_tiles"
SCAN_LIST="scan_list.txt"

# Build the scan list from subdirectories of the scan data folder
ls -d "${SCAN_DIR}/"*/ 2>/dev/null | xargs -n1 basename | sort > "${SCAN_LIST}"

N=$(wc -l < "${SCAN_LIST}")

if [ "${N}" -eq 0 ]; then
    echo "ERROR: No scan folders found in ${SCAN_DIR}"
    exit 1
fi

echo "Found ${N} scans in ${SCAN_DIR}"
echo "Scan list written to ${SCAN_LIST}"

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Submitting array job for ${N} scans (array indices 0-$((N - 1)))..."
sbatch --array=0-$((N - 1)) run_pipeline_array.sh

echo "Done. Monitor with: squeue -u \$USER"
