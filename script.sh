#!/bin/bash
HOST=localhost
PORT=8787
MAX_TOKENS=4096
LANGUAGE=arb_Arab
MODEL_NAME=$1
GPU_COUNT=$2
WORK_DIR="/ibex/ai/home/alyafez/JQL-Annotation-Pipeline"
VENV_PATH="/ibex/ai/home/alyafez/.vllm/bin/activate"
cd ${WORK_DIR}
source ${VENV_PATH}
CMD="
    vllm serve \
        ${MODEL_NAME} \
        --host ${HOST} \
        --port ${PORT} \
        --gpu-memory-utilization 0.95 \
        --tensor-parallel-size ${GPU_COUNT} \
        --pipeline-parallel-size 1 \
        --max-model-len ${MAX_TOKENS} \
        --max-num-seqs 100 \
        --limit-mm-per-prompt image=0 \
        --dtype=bfloat16 \
	    --served-model-name ${MODEL_NAME} 2>&1 > /dev/null
    "
echo "Starting VLLM server on ${HOST}:${PORT}..."
bash -c "${CMD}" & python score.py --mode vllm-online --model ${MODEL_NAME} --language ${LANGUAGE} --num-examples 50000 --batch-size 50 --max-tokens ${MAX_TOKENS}
bash killall.sh
exit 0
