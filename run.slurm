#!/bin/bash
#SBATCH --job-name=jql-annotation
#SBATCH --output=slurm/jql-annotation_%A_%a.out
#SBATCH --error=slurm/jql-annotation_%A_%a.err
#SBATCH --partition=batch
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=16GB
#SBATCH --time=4:00:00
#SBATCH --array=0-139%8

MODEL_NAME="Qwen3-8B"
GPU_COUNT=2
WORK_DIR="/ibex/ai/home/alyafez/JQL-Annotation-Pipeline"
VENV_PATH="/ibex/ai/home/alyafez/.vllm/bin/activate"

cd ${WORK_DIR}
source ${VENV_PATH}

bash script.sh ${MODEL_NAME} ${GPU_COUNT} ${SLURM_ARRAY_TASK_ID} ${SLURM_ARRAY_TASK_COUNT}