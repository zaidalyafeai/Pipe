#!/bin/bash
MODEL_NAME="gemma-3-12b-it"
GPU_COUNT=2
WORK_DIR="/ibex/ai/home/alyafez/JQL-Annotation-Pipeline"
VENV_PATH="/ibex/ai/home/alyafez/.vllm/bin/activate"

# Submit the job
sbatch --job-name=${MODEL_NAME} \
       --gres=gpu:a100:${GPU_COUNT} \
       --time=6:00:00 \
       --cpus-per-gpu=6 \
       --partition=batch \
       --output=slurm/${MODEL_NAME}_%A_%a.out \
       --error=slurm/${MODEL_NAME}_%A_%a.err \
        script.sh ${MODEL_NAME} ${GPU_COUNT}