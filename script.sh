HOST=localhost
PORT=8787
MODEL_NAME=gemma-3-4b-it
MODEL_PATH=${MODEL_NAME}
MAX_TOKENS=4096
LANGUAGE=arb_Arab
CMD="
    vllm serve \
        ${MODEL_PATH} \
        --host ${HOST} \
        --port ${PORT} \
        --gpu-memory-utilization 0.95 \
        --tensor-parallel-size $1 \
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
