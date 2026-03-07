#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=run_segmentation_mean_teacher_joint_pseudo_label.out
#SBATCH --error=run_segmentation_mean_teacher_joint_pseudo_label.err
#SBATCH --mail-type=END

module use /opt/insy/modulefiles
module load cuda/12.1

source ~/.bashrc
conda activate /home/nfs/ccharlesworth/segmentation_conda

# Exp 2 + Exp 6: Joint training from scratch AND pseudo-label mode.
# No separate Phase 1. Supervised + pseudo-label loss run jointly from epoch 0.
# The 50-epoch rampup lets the teacher develop before its pseudo-labels are
# trusted — at epoch 0 with random weights, mean confidence is ~0.5 so
# virtually no crops will exceed the 0.8 threshold, making the rampup implicit.
# As the teacher improves, more crops become confident and contribute.
# consistency_weight=10.0 since pseudo-label loss is on the same scale as
# the supervised loss.

echo "Starting Mean Teacher training (Exp 2+6: joint training + pseudo-labels)"

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
    --pseudo_label_threshold 0.8
