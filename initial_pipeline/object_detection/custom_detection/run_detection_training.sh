#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=1:45:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --output=run_detection_training.out
#SBATCH --error=run_detection_training.err
#SBATCH --mail-type=END

module use /opt/insy/modulefiles
module load cuda/12.1

source ~/.bashrc   
conda activate /home/nfs/ccharlesworth/segmentation_conda

echo "starting job file"
srun python -u yolo_training.py