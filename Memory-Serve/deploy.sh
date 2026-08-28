export VLLM_USE_FLASHINFER_SAMPLER=0
CUDA_VISIBLE_DEVICES=4,5,6,7 \
vllm serve /path/to/Qwen3-32B \
    --tensor-parallel-size 4 \
    --served-model-name "qwen" \
    --gpu-memory-utilization 0.8 \
    --host 0.0.0.0 \
    --port 8002 \
    --trust-remote-code
