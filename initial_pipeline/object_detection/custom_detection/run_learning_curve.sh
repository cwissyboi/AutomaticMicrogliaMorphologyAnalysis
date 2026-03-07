#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=2:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=run_learning_curve.out
#SBATCH --error=run_learning_curve.err
#SBATCH --mail-type=END

module use /opt/insy/modulefiles
module load cuda/12.1

source ~/.bashrc
conda activate /home/nfs/ccharlesworth/segmentation_conda

echo "starting learning curve job"
srun python -u yolo_learning_curve.py
