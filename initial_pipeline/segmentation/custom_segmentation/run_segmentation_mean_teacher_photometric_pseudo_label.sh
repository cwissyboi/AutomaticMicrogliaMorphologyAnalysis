#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=1:45:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=run_segmentation_mean_teacher_photometric_pseudo_label.out
#SBATCH --error=run_segmentation_mean_teacher_photometric_pseudo_label.err
#SBATCH --mail-type=END

module use /opt/insy/modulefiles
module load cuda/12.1

source ~/.bashrc
conda activate /home/nfs/ccharlesworth/segmentation_conda

# Exp 4 + Exp 6: Photometric augmentation AND pseudo-label mode.
# Phase 1 supervised pre-training runs as normal, then Phase 2 uses the teacher
# to produce hard pseudo-labels on photometrically-perturbed unlabelled crops.
# The photometric augmentation makes the teacher's job harder (noisy inputs),
# which should only keep confident crops above the threshold — filtering out
# ambiguous predictions and yielding cleaner pseudo-labels.
# consistency_weight=1.0 is appropriate since pseudo-label loss is on the same
# scale as the supervised loss (BCE+clDice, not tiny MSE).

echo "Starting Mean Teacher training (Exp 4+6: photometric augmentation + pseudo-labels)"

srun python mean_teacher_training.py \
    --unlabelled_dir "../../../../SegmentationDatasets/UnlabelledCells/" \
    --loss_type bce_cldice \
    --cldice_alpha 0.5 \
    --ema_alpha 0.999 \
    --consistency_weight 10.0 \
    --consistency_rampup 0 \
    --unlabelled_batch_size 16 \
    --epochs 1 \
    --patience 1 \
    --finetune_epochs 100 \
    --finetune_patience 10 \
    --finetune_lr 1e-4 \
    --photometric_augmentation \
    --pseudo_label_threshold 0.8
