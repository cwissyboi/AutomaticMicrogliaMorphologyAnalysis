#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=5:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --output=run_segmentation_mean_teacher_photometric_joint.out
#SBATCH --error=run_segmentation_mean_teacher_photometric_joint.err
#SBATCH --mail-type=END

module use /opt/insy/modulefiles
module load cuda/12.1

source ~/.bashrc
conda activate /home/nfs/ccharlesworth/segmentation_conda

# Exp 4 + Exp 2: Photometric augmentation AND joint training from scratch.
# No separate Phase 1. Supervised + MT consistency trained jointly from epoch 0
# with a 50-epoch rampup so the supervised signal dominates early.
# Photometric perturbations (ColorJitter, GaussianBlur, GaussNoise) on unlabelled
# data give a genuine invariance signal beyond geometry.
# lr=1e-4 throughout (same as Phase 1 in the standard run).

echo "Starting Mean Teacher training (Exp 4+2: photometric augmentation + joint training)"

srun python mean_teacher_training.py \
    --unlabelled_dir "../../../../SegmentationDatasets/UnlabelledCells/" \
    --loss_type bce_cldice \
    --cldice_alpha 0.5 \
    --ema_alpha 0.999 \
    --consistency_weight 2.0 \
    --consistency_rampup 20 \
    --unlabelled_batch_size 16 \
    --joint_training \
    --finetune_epochs 200 \
    --finetune_patience 30 \
    --finetune_lr 1e-4 \
    --photometric_augmentation
