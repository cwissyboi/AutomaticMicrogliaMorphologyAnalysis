#!/bin/bash
#SBATCH --job-name=microglia_pipeline
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/pipeline_%a.out
#SBATCH --error=logs/pipeline_%a.err
#SBATCH --mail-type=FAIL,ARRAY_TASKS

# run_pipeline_array.sh
#
# SLURM array job: processes one scan per task.
# Submit via submit_pipeline.sh (do not submit this file directly).
#
# Each task reads its scan name from scan_list.txt using SLURM_ARRAY_TASK_ID
# as the line index, then runs main.py for that scan.

set -euo pipefail

module use /opt/insy/modulefiles
module load cuda/12.1

source ~/.bashrc
conda activate /home/nfs/ccharlesworth/segmentation_conda

# Resolve the scan name for this array task (1-indexed line in scan_list.txt)
SCAN_LIST="scan_list.txt"
SCAN_NAME=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${SCAN_LIST}")

if [ -z "${SCAN_NAME}" ]; then
    echo "ERROR: Could not read scan name for array task ${SLURM_ARRAY_TASK_ID} from ${SCAN_LIST}"
    exit 1
fi

INPUT_PATH="../../ScanData\10_patients_GM_tiles/${SCAN_NAME}"

echo "Array task : ${SLURM_ARRAY_TASK_ID}"
echo "Scan name  : ${SCAN_NAME}"
echo "Input path : ${INPUT_PATH}"
echo "Node       : $(hostname)"
echo "Start time : $(date)"

# cd /tudelft.net/staff-umbrella/AutomaticMicrogliaMorphologyAnalysis/initial_pipeline

srun python -u main.py \
    --input_folder_path "${INPUT_PATH}" \
    --output_name pipeline_run \
    --scan_name "${SCAN_NAME}"

echo "Finished    : $(date)"
