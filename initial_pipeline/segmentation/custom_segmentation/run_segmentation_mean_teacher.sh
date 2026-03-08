#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=5:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --output=run_segmentation_mean_teacher.out
#SBATCH --error=run_segmentation_mean_teacher.err
#SBATCH --mail-type=END

module use /opt/insy/modulefiles
module load cuda/12.1

source ~/.bashrc
conda activate /home/nfs/ccharlesworth/segmentation_conda

# Path to the unlabelled cell crops produced by YOLO detection.
# Each sub-folder under this directory corresponds to one scan.

echo "Starting Mean Teacher semi-supervised training"

srun python mean_teacher_training.py \
    --unlabelled_dir "../../../../SegmentationDatasets/UnlabelledCells/" \
    --loss_type bce_cldice \
    --cldice_alpha 0.5 \
    --ema_alpha 0.999 \
    --consistency_weight 2.0 \
    --consistency_rampup 0 \
    --unlabelled_batch_size 16 \
    --epochs 100 \
    --patience 10 \
    --finetune_epochs 20 \
    --finetune_patience 30 \
    --finetune_lr 1e-5
