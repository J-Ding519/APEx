   
export AGENT_URL="http://localhost:8000/v1"

export SERVICE_URL="http://127.0.0.1:8001/"

export TEST_CACHE_DIR="/path/to/data/image_search_cache/fvqa_train_cache, /path/to/data/image_search_cache/fvqa_test_cache"
export MAX_LLM_CALL_PER_RUN=20

python agent_serve.py