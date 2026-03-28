#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=2:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=run_detection_training_additional_al_as_testing.out
#SBATCH --error=run_detection_training_additional_al_as_testing.err
#SBATCH --mail-type=END

module use /opt/insy/modulefiles
module load cuda/12.1

source ~/.bashrc   
conda activate /home/nfs/ccharlesworth/segmentation_conda

echo "starting job file"
srun python -u yolo_training_single_split.py --test-data-yaml data_additional_al.yaml