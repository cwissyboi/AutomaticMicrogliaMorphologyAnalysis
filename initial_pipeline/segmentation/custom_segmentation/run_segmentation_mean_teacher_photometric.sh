#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=1:45:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=run_segmentation_mean_teacher_photometric.out
#SBATCH --error=run_segmentation_mean_teacher_photometric.err
#SBATCH --mail-type=END

module use /opt/insy/modulefiles
module load cuda/12.1

source ~/.bashrc
conda activate /home/nfs/ccharlesworth/segmentation_conda

# Exp 4: Photometric augmentation on unlabelled data.
# Adds ColorJitter, GaussianBlur and GaussNoise to the unlabelled transform
# pipeline. These are NOT inverted before the consistency loss, so the student
# must produce predictions that are invariant to appearance changes — giving a
# genuine training signal beyond the geometric-only baseline.
# All other hyperparameters match the standard run.

echo "Starting Mean Teacher semi-supervised training (Exp 4: photometric augmentation)"

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
    --photometric_augmentation
