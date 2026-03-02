#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=run_yolo_inference.out
#SBATCH --error=run_yolo_inference.err
#SBATCH --mail-type=END

module use /opt/insy/modulefiles
module load cuda/12.1

source ~/.bashrc   
conda activate /home/nfs/ccharlesworth/segmentation_conda

echo "starting job file"
srun python yolo_inference_only.py --input_folder_path ../../../All_scans_tiled --output_name yolo_for_semi_supervised