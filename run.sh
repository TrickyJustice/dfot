#!/bin/bash
#SBATCH --account=def-sirisha
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5    # There are 24 CPU cores on P100 Cedar GPU nodes
#SBATCH --mem=42GB              # Request the full memory of the node
#SBATCH --time=0-00:10:00

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo “Hello World”
nvidia-smi

#Load needed python and cuda modules
module load python opencv

#Activate your environment
source dfot/bin/activate

#Variables for readability
# logdir=/u1/a2soni/stable-diffusion/saved

python -m main +name=single_image_to_short dataset=realestate10k_mini algorithm=dfot_video_pose experiment=video_generation @diffusion/continuous load=pretrained:DFoT_RE10K.ckpt 'experiment.tasks=[validation]' experiment.validation.data.shuffle=True dataset.context_length=1 dataset.frame_skip=20 dataset.n_frames=8 experiment.validation.batch_size=1 algorithm.tasks.prediction.history_guidance.name=vanilla +algorithm.tasks.prediction.history_guidance.guidance_scale=4.0
