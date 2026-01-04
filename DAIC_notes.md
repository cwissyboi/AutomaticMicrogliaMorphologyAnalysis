
Connect via ssh using 
ssh ccharlesworth@login.daic.tudelft.nl


Copy a file from computer to DAIC
$ scp make_env_job.sh ccharlesworth@login.daic.tudelft.nl:~/test_code/

Copy a folder from computer to DAIC
scp -r uranage1.gif ccharlesworth@login.daic.tudelft.nl:~/

Remove a file 
rm filename


Copy a file from DAIC to computer 
scp [<YourNetID>@]login.daic.tudelft.nl:~/origin_path_on_DAIC/remotefile ./

Copy a folder from DAIC to computer 
scp -r [<YourNetID>@]login.daic.tudelft.nl:~/origin_path_on_DAIC/remotefolder ./

Home directory /home/nfs/ccharlesworth

submit job
sbatch job_file.sh
- submit a batch script
squeue - check the status of jobs on the system
scancel - cancel a job and delete it from the queue



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




DELFT BLUE 
ssh ccharlesworth@login.delftblue.tudelft.nl