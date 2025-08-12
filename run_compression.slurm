#!/bin/bash
#SBATCH --job-name=run_compression
#SBATCH --time=0:05:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=6
#SBATCH --partition=batch
#SBATCH --output=slurm/run_compression_%A_%a.out
#SBATCH --error=slurm/run_compression_%A_%a.err
#SBATCH --array=0-139%32

# Load any necessary modules here
# module load cuda/11.8

# Change to the script directory
cd /ibex/ai/home/alyafez/JQL-Annotation-Pipeline
source /ibex/ai/home/alyafez/.jql/bin/activate

# Run the script
python run_compression.py  --shard ${SLURM_ARRAY_TASK_ID} --num_shards 140