#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --output=run_segmentation_mean_teacher_joint_w_5.out
#SBATCH --error=run_segmentation_mean_teacher_joint_w_5.err
#SBATCH --mail-type=END

module use /opt/insy/modulefiles
module load cuda/12.1

source ~/.bashrc
conda activate /home/nfs/ccharlesworth/segmentation_conda

# Exp 2: Joint training from scratch (no separate Phase 1).
# Skips the pure supervised pre-training phase and runs supervised + Mean
# Teacher consistency jointly from epoch 0. The consistency rampup (50 epochs)
# ensures the supervised signal dominates early so the teacher can develop
# meaningful predictions before the MT term contributes significantly.
# --finetune_lr is used as the single learning rate throughout.
# --finetune_patience is generous (30) to accommodate the noisier MT signal.
# consistency_weight 10 keeps the unlabelled term meaningful but not dominant.

echo "Starting Mean Teacher semi-supervised training (Exp 2: joint training from scratch)"

srun python mean_teacher_training.py \
    --unlabelled_dir "../../../../SegmentationDatasets/UnlabelledCells/" \
    --loss_type bce_cldice \
    --cldice_alpha 0.5 \
    --ema_alpha 0.999 \
    --consistency_weight 5 \
    --consistency_rampup 20 \
    --unlabelled_batch_size 16 \
    --joint_training \
    --finetune_epochs 200 \
    --finetune_patience 30 \
    --finetune_lr 1e-4
