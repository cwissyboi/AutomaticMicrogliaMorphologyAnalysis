#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=2:45:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=run_segmentation_mean_teacher_no_pretraining_weight_10.out
#SBATCH --error=run_segmentation_mean_teacher_no_pretraining_weight_10.err
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
    --consistency_weight 10.0 \
    --consistency_rampup 0 \
    --unlabelled_batch_size 16 \
    --epochs 1 \
    --patience 1 \
    --finetune_epochs 100 \
    --finetune_patience 10 \
    --finetune_lr 1e-5
