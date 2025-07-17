#!/bin/bash
#SBATCH --job-name=run_classifier
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:45:00
#SBATCH --cpus-per-gpu=6
#SBATCH --ntasks-per-node=1
#SBATCH --partition=batch
#SBATCH --output=slurm/run_classifier_%A_%a.out
#SBATCH --error=slurm/run_classifier_%A_%a.err
#SBATCH --array=0-7%2

# Load any necessary modules here
# module load cuda/11.8

# Change to the script directory
cd /ibex/ai/home/alyafez/JQL-Annotation-Pipeline
source /ibex/ai/home/alyafez/.jql/bin/activate

# Run the script
python run_classifier.py  --shard ${SLURM_ARRAY_TASK_ID} --num_shards 8
