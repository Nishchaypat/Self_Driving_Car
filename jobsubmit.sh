#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128gb
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sshrestha8@gsu.edu
#SBATCH --account=aas3s2
#SBATCH --partition=qGPU48
#SBATCH --gres=gpu:2
#SBATCH --output=outputs/output_%j
#SBATCH --error=errors/error_%j

cd /scratch
mkdir $SLURM_JOB_ID
cd $SLURM_JOB_ID


source /userapp/virtualenv/torch_ssd_env/venv/bin/activate


python -u train_ssd.py
