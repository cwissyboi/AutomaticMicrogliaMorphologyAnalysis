#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --output=run_segmentation_mean_teacher_joint_pseudo_label_frac50_w0.out
#SBATCH --error=run_segmentation_mean_teacher_joint_pseudo_label_frac50_w0.err
#SBATCH --mail-type=END

module use /opt/insy/modulefiles
module load cuda/12.1

source ~/.bashrc
conda activate /home/nfs/ccharlesworth/segmentation_conda

# Labelled-fraction ablation: 50% of labelled training data.
# All other settings match run_segmentation_mean_teacher_joint_pseudo_label.sh
# (old_runs/). Use this together with the 10/25/75/100 variants to assess
# whether Mean Teacher provides benefit at low labelled-data regimes.

echo "Starting Mean Teacher training (joint + pseudo-labels, 50% labelled data)"

srun python mean_teacher_training.py \
    --unlabelled_dir "../../../../SegmentationDatasets/UnlabelledCells/" \
    --loss_type bce_cldice \
    --cldice_alpha 0.5 \
    --ema_alpha 0.999 \
    --consistency_weight 0 \
    --consistency_rampup 20 \
    --unlabelled_batch_size 16 \
    --joint_training \
    --finetune_epochs 200 \
    --finetune_patience 30 \
    --finetune_lr 1e-4 \
    --pseudo_label_threshold 0.8 \
    --labelled_fraction 0.5
