DATASET=/path/to/data/Train/Executor/fvqa_train.jsonl

OUTPUT_PATH=/path/to/results/mmsearch-planner/result/fvqa_train

MODEL_PATH=/path/to/checkpoints/Trained-Executor

export MAX_REFLECTION_TIMES=1

export SERVICE_URL="http://127.0.0.1:8003/"

export TEST_CACHE_DIR="/path/to/data/image_search_cache/fvqa_train_cache, /path/to/data/image_search_cache/fvqa_test_cache"

export PLAN_URL="http://127.0.0.1:5000/plan"

export REPLAN_JUDGE_URL="http://127.0.0.1:5000/judge_replan"

export REPLAN_URL="http://127.0.0.1:5000/replan"

export MEMORY_BASE_URL="http://127.0.0.1:5000"

export MAX_LLM_CALL_PER_RUN=30

export JUDGE_URL="http://127.0.0.1:8002/v1"

export JUDGE_PROMPT_TYPE="default"

python -u run_multi_react.py \
    --dataset "$DATASET" \
    --output "$OUTPUT_PATH" \
    --max_workers 2 \
    --max_tokens 10000\
    --model $MODEL_PATH \
    --temperature 0 \
    --presence_penalty 1.1 \
    --top_p 1.0 \
    --roll_out_count 1 \
    --main_ports "8000"

