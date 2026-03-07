#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=1:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --output=run_segmentation_disconnect_components.out
#SBATCH --error=run_segmentation_disconnect_components.err
#SBATCH --mail-type=END

module use /opt/insy/modulefiles
module load cuda/12.1

source ~/.bashrc   
conda activate /home/nfs/ccharlesworth/segmentation_conda

echo "starting job file"
srun python -u segmentation_training.py --disconnect_components True --loss_type bce_cldice --cldice_alpha 0.5
