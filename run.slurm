#!/bin/bash
#SBATCH --job-name=vllm_serve
#SBATCH --gres=gpu:a100:2
#SBATCH --time=1:00:00
#SBATCH --cpus-per-gpu=6
#SBATCH --partition=batch
#SBATCH --output=slurm/vllm_serve_%A_%a.out
#SBATCH --error=slurm/vllm_serve_%A_%a.err
#SBATCH --array=1-10%1

# Load any necessary modules here
# module load cuda/11.8

# Change to the script directory
/ibex/ai/home/alyafez/JQL-Annotation-Pipeline
source /ibex/ai/home/alyafez/.vllm/bin/activate

# Run the script
bash script.sh 2