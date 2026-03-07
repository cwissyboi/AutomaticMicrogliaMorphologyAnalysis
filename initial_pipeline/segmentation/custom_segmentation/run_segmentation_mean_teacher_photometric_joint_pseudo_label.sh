#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=run_segmentation_mean_teacher_photometric_joint_pseudo_label.out
#SBATCH --error=run_segmentation_mean_teacher_photometric_joint_pseudo_label.err
#SBATCH --mail-type=END

module use /opt/insy/modulefiles
module load cuda/12.1

source ~/.bashrc
conda activate /home/nfs/ccharlesworth/segmentation_conda

# Exp 4 + Exp 2 + Exp 6: Full combination — photometric augmentation,
# joint training from scratch, AND pseudo-label mode.
# No separate Phase 1. Photometric perturbations on unlabelled data mean the
# teacher must be genuinely robust to appearance variation before its predictions
# pass the 0.8 confidence threshold. This acts as a natural quality gate:
# only crops the teacher is very sure about (despite colour/blur/noise) become
# pseudo-labels. The 50-epoch rampup gives the teacher time to develop before
# its pseudo-labels meaningfully contribute. lr=1e-4 throughout.

echo "Starting Mean Teacher training (Exp 4+2+6: photometric + joint + pseudo-labels)"

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
    --photometric_augmentation \
    --pseudo_label_threshold 0.8
