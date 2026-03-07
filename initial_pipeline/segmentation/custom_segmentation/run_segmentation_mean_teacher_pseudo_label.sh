#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=1:45:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=run_segmentation_mean_teacher_pseudo_label.out
#SBATCH --error=run_segmentation_mean_teacher_pseudo_label.err
#SBATCH --mail-type=END

module use /opt/insy/modulefiles
module load cuda/12.1

source ~/.bashrc
conda activate /home/nfs/ccharlesworth/segmentation_conda

# Exp 6: Pseudo-label mode (replaces soft MSE consistency).
# Teacher predictions with mean pixel confidence >= 0.8 are binarised and
# used as hard pseudo-ground-truth masks for the student's supervised loss.
# Crops below the threshold are skipped entirely.
# This avoids the small-magnitude MSE gradient problem by using the same
# supervised loss (BCE+clDice) on confidently-predicted unlabelled crops.
# consistency_weight scales the pseudo-label loss relative to the supervised
# loss on labelled data.
# All other hyperparameters match the standard run.

echo "Starting Mean Teacher semi-supervised training (Exp 6: pseudo-label mode)"

srun python mean_teacher_training.py \
    --unlabelled_dir "../../../../SegmentationDatasets/UnlabelledCells/" \
    --loss_type bce_cldice \
    --cldice_alpha 0.5 \
    --ema_alpha 0.999 \
    --consistency_weight 1.0 \
    --consistency_rampup 0 \
    --unlabelled_batch_size 16 \
    --epochs 1 \
    --patience 1 \
    --finetune_epochs 100 \
    --finetune_patience 10 \
    --finetune_lr 1e-4 \
    --pseudo_label_threshold 0.8
