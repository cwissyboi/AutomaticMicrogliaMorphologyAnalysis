
Connect via ssh using 
ssh ccharlesworth@login.daic.tudelft.nl

cd /tudelft.net/staff-umbrella/AutomaticMicrogliaMorphologyAnalysis

cd /tudelft.net/staff-umbrella/StudentsCVlab/


Copy a file from computer to DAIC
$ scp make_env_job.sh ccharlesworth@login.daic.tudelft.nl:~/test_code/

Copy a folder from computer to DAIC
scp -r uranage1.gif ccharlesworth@login.daic.tudelft.nl:~/


scp -r AnnotationsData_Adjusted_WithSoma ccharlesworth@login.daic.tudelft.nl:/tudelft.net/staff-umbrella/AutomaticMicrogliaMorphologyAnalysis/

scp 28_2_rat_pretraining.pt ccharlesworth@login.daic.tudelft.nl:/tudelft.net/staff-umbrella/AutomaticMicrogliaMorphologyAnalysis/AutomaticMicrogliaMorphologyAnalysis/initial_pipeline/object_detection/custom_detection/yolo_good_runs

scp best_run_25_1.pth ccharlesworth@login.daic.tudelft.nl:/tudelft.net/staff-umbrella/AutomaticMicrogliaMorphologyAnalysis/AutomaticMicrogliaMorphologyAnalysis/initial_pipeline/segmentation/custom_segmentation/checkpoints

scp All_scans_AD.zip ccharlesworth@login.daic.tudelft.nl:/tudelft.net/staff-umbrella/AutomaticMicrogliaMorphologyAnalysis/ScanData/

scp -r YOLO_rat_adjusted_2_split ccharlesworth@login.daic.tudelft.nl:/tudelft.net/staff-umbrella/AutomaticMicrogliaMorphologyAnalysis/

scp -r YOLO_dataset_adjusted_2 ccharlesworth@login.daic.tudelft.nl:/tudelft.net/staff-umbrella/AutomaticMicrogliaMorphologyAnalysis/

scp -r YOLO_dataset_with_al ccharlesworth@login.daic.tudelft.nl:/tudelft.net/staff-umbrella/AutomaticMicrogliaMorphologyAnalysis/DetectionDatasets/

scp -r YOLO_dataset_with_edge_confidence ccharlesworth@login.daic.tudelft.nl:/tudelft.net/staff-umbrella/AutomaticMicrogliaMorphologyAnalysis/DetectionDatasets/

scp -r YOLO_dataset_with_random ccharlesworth@login.daic.tudelft.nl:/tudelft.net/staff-umbrella/AutomaticMicrogliaMorphologyAnalysis/DetectionDatasets/
scp -r All_scans_tiled ccharlesworth@login.daic.tudelft.nl:/tudelft.net/staff-umbrella/AutomaticMicrogliaMorphologyAnalysis/

scp -r custom_segmentation ccharlesworth@login.daic.tudelft.nl:/tudelft.net/staff-umbrella/AutomaticMicrogliaMorphologyAnalysis/initial_pipeline/segmentation/
scp run_segmentation_cldice.sh ccharlesworth@login.daic.tudelft.nl:/tudelft.net/staff-umbrella/AutomaticMicrogliaMorphologyAnalysis/initial_pipeline/segmentation/custom_segmentation


sacctmgr show user ccharlesworth withassoc Format='DefaultAccount,Account' --parsable # Check your account(s)

scp -r custom_detection ccharlesworth@login.daic.tudelft.nl:/tudelft.net/staff-umbrella/AutomaticMicrogliaMorphologyAnalysis/initial_pipeline/object_detection/

scp -r YOLO_dataset_adjusted ccharlesworth@login.daic.tudelft.nl:/tudelft.net/staff-umbrella/AutomaticMicrogliaMorphologyAnalysis/
scp -r YOLO_dataset_adjusted_gray ccharlesworth@login.daic.tudelft.nl:/tudelft.net/staff-umbrella/AutomaticMicrogliaMorphologyAnalysis/

scp main.py ccharlesworth@login.daic.tudelft.nl:/tudelft.net/staff-umbrella/AutomaticMicrogliaMorphologyAnalysis/initial_pipeline/

scp segmentation_requirements.in ccharlesworth@login.daic.tudelft.nl:~/

/tudelft.net/staff-umbrella/StudentsCVlab/

Remove a file 
rm filename


python convert_coco_to_annotations.py --input SegmentationAnnotationsAdjusted --output AnnotationsData_Adjusted/Segmentations

Copy a file from DAIC to computer 
scp [<YourNetID>@]login.daic.tudelft.nl:~/origin_path_on_DAIC/remotefile ./

Copy a folder from DAIC to computer 
scp -r [<YourNetID>@]login.daic.tudelft.nl:~/origin_path_on_DAIC/remotefolder ./


scp -r ccharlesworth@login.daic.tudelft.nl:/tudelft.net/staff-umbrella/AutomaticMicrogliaMorphologyAnalysis/AutomaticMicrogliaMorphologyAnalysis/initial_pipeline/morphology/morphology_outputs ./


Home directory /home/nfs/ccharlesworth

submit job
sbatch job_file.sh
- submit a batch script
squeue - check the status of jobs on the system
scancel - cancel a job and delete it from the queue

scp ccharlesworth@login.daic.tudelft.nl:/tudelft.net/staff-umbrella/AutomaticMicrogliaMorphologyAnalysis/initial_pipeline/segmentation/custom_segmentation/checkpoints/best_run_25_1.pth ./

scp ccharlesworth@login.daic.tudelft.nl:/tudelft.net/staff-umbrella/AutomaticMicrogliaMorphologyAnalysis/AutomaticMicrogliaMorphologyAnalysis/initial_pipeline/object_detection/custom_detection/yolo_good_runs/28_2_rat_pretraining.pt ./

Links 
https://daic.tudelft.nl/docs/manual/connecting/
https://daic.tudelft.nl/docs/manual/job-submission/slurm-basics/#interactive-jobs-on-compute-nodes
https://daic.tudelft.nl/docs/manual/software/modules/



Random linux commands 
Number of files in my current dir 
ls -1 | wc -l

delete cache directory
rm -rf ~/.cache

Size of files in this directory 
du -h --max-depth=1 ~ | sort -hr

Conda commands 
conda env create -f .\conda_test_env.yml  

get all conda environments
conda env list
conda info --envs

conda activate env_name
conda activate /home/nfs/ccharlesworth/myenv

pip install without using a cache
pip install --no-cache-dir pandas

pip install --no-cache-dir -r initial_pipeline_requirements.txt

Create conda environment in the home directory
module use /opt/insy/modulefiles
module load miniconda/3.11
conda create --prefix ./segmentation_venv python=3.10
conda create --prefix ./segmentation_conda python=3.10


pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
before pip installing the requirements


conda activate ./segmentation_conda


module load miniconda/3.11
conda create -n segmentation_venv python=3.10 pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -c conda-forge
conda activate segmentation_venv

conda create --prefix ./segmentation_conda python=3.10



DELFT BLUE 
ssh ccharlesworth@login.delftblue.tudelft.nl

/home/nfs/ccharlesworth/segmentation_conda


no deps means only install this.
pip install pandas==2.1.4 --no-deps


pip install pand==2.1.4 --no-deps

python segmentation_training.py --add_new_components True --disconnect_components False


numpy 1.26.4


sacct -u $USER --format=JobID,JobName%100,Partition,State,Elapsed,Start,End


sacct -j 12254770  --format=JobID,JobName,MaxRSS,ReqMem



sacct -j 12254770 --format=JobID,JobName%100

run_segmentation_mean_teacher_joint.sh 
run_segmentation_mean_teacher_joint_pseudo_label_w_1.sh 
run_segmentation_mean_teacher_joint_pseudo_label_w_5.sh 

python yolo_find_threshold_annotations.py --input_folder_path ../../../../ScanData/All_scans_tiled --output_name edge_confidence_cells --max_images 487

python yolo_find_random_annotations.py --input_folder_path ../../../../ScanData/All_scans_tiled --output_name random_cells --max_images 487