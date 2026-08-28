#!/bin/bash
# ============================================================
# 单卡调试脚本：用 Qwen3-8B 同时充当 Executor + Planner + Judger
# 仅跑 5 条样本，验证整条流水线是否通畅
# ============================================================

set -e

# ======================== 路径配置 ========================
BASE_DIR="/path/to/workdir"
CODE_DIR="${BASE_DIR}/MIA-private"
DATA_DIR="${BASE_DIR}/MIA-data"
OTHER_DIR="${BASE_DIR}/MIA-other"
SEARCH_R1_DIR="${BASE_DIR}/Search-R1"
E5_MODEL="${OTHER_DIR}/e5-base-v2"
FAISS_INDEX="${OTHER_DIR}/faiss_index/e5_Flat.index"
WIKI25_MERGED="${OTHER_DIR}/wiki25_merged/wiki25.jsonl"
LOG_DIR="${BASE_DIR}/logs"

DEBUG_MODEL="/path/to/Qwen2.5-VL-7B-Instruct"
DEBUG_DATASET="${DATA_DIR}/Train/Executor/fvqa_train.jsonl"
DEBUG_OUTPUT="${BASE_DIR}/results/debug_test"
DEBUG_DATASET_SMALL="${DEBUG_OUTPUT}/debug_5samples.jsonl"

mkdir -p "${LOG_DIR}" "${DEBUG_OUTPUT}"

# ======================== 辅助函数 ========================
wait_for_service() {
    local url=$1
    local name=$2
    local max_wait=$3
    local waited=0
    echo "[等待] ${name} 启动中..."
    while ! curl -s "${url}" > /dev/null 2>&1; do
        sleep 10
        waited=$((waited + 10))
        if [ ${waited} -ge ${max_wait} ]; then
            echo "[超时] ${name} 在 ${max_wait}s 内未就绪，请检查 ${LOG_DIR}/"
            exit 1
        fi
        echo "  ... 已等待 ${waited}s"
    done
    echo "[就绪] ${name} 已启动 (${waited}s)"
}

cleanup_services() {
    echo "清理所有服务..."
    pkill -f "vllm serve" 2>/dev/null || true
    pkill -f "memory_serve.py" 2>/dev/null || true
    pkill -f "retrieval_server.py" 2>/dev/null || true
    sleep 3
}

# ======================== Step 0: 准备 ========================
echo "========== [调试模式] 单卡 Qwen3-8B =========="

cleanup_services

# 截取前5条样本
echo "截取前5条样本..."
head -n 5 "${DEBUG_DATASET}" > "${DEBUG_DATASET_SMALL}"
echo "样本数: $(wc -l < ${DEBUG_DATASET_SMALL})"

# ======================== Step 1: 启动 Qwen3-8B (一个模型充当所有角色) ========================
echo "========== [Step 1] 启动 Qwen3-8B :8000 (Executor + Planner + Judger) =========="
export VLLM_USE_FLASHINFER_SAMPLER=0
CUDA_VISIBLE_DEVICES=0 \
nohup vllm serve "${DEBUG_MODEL}" \
    --tensor-parallel-size 1 \
    --served-model-name "qwen" \
    --gpu-memory-utilization 0.8 \
    --enforce-eager \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    > "${LOG_DIR}/debug_vllm.log" 2>&1 &
VLLM_PID=$!
echo "vLLM PID: ${VLLM_PID}"

# ======================== Step 2: 启动 Search-R1 ========================
echo "========== [Step 2] 启动 Search-R1 :8003 =========="
cd "${SEARCH_R1_DIR}"
sed -i 's/port=[0-9]\+/port=8003/g' search_r1/search/retrieval_server.py 2>/dev/null || true
nohup python search_r1/search/retrieval_server.py \
    --index_path "${FAISS_INDEX}" \
    --corpus_path "${WIKI25_MERGED}" \
    --topk 3 \
    --retriever_name e5 \
    --retriever_model "${E5_MODEL}" \
    > "${LOG_DIR}/debug_search.log" 2>&1 &
SEARCH_PID=$!
cd "${CODE_DIR}"
echo "Search PID: ${SEARCH_PID}"

# ======================== 等待 vLLM 就绪 ========================
wait_for_service "http://localhost:8000/v1/models" "Qwen3-8B (8000)" 300

# ======================== Step 3: 启动 Memory-Serve ========================
echo "========== [Step 3] 启动 Memory-Serve :5000 =========="
# 全部指向同一个 Qwen3-8B :8000
export MEMORY_URL="http://localhost:8000/v1"
export PLAN_URL="http://localhost:8000/v1"
cd "${CODE_DIR}/Memory-Serve"
nohup python memory_serve.py \
    --model_name qwen \
    --port 5000 \
    --host 0.0.0.0 \
    > "${LOG_DIR}/debug_memory.log" 2>&1 &
MEMORY_PID=$!
echo "Memory PID: ${MEMORY_PID}"

# ======================== 等待所有服务就绪 ========================
wait_for_service "http://localhost:5000/hallo" "Memory Service (5000)" 120
wait_for_service "http://localhost:8003/retrieve" "Search (8003)" 600

echo ""
echo "=========================================="
echo "  所有服务已就绪！(调试模式)"
echo "  Qwen3-8B:  PID=${VLLM_PID}   port=8000 (全部角色)"
echo "  Search:    PID=${SEARCH_PID}   port=8003"
echo "  Memory:    PID=${MEMORY_PID}   port=5000"
echo "=========================================="
echo ""

# ======================== 推理环境变量 ========================
export SERVICE_URL="http://localhost:8003/"
export PLAN_URL="http://localhost:5000/plan"
export REPLAN_JUDGE_URL="http://localhost:5000/judge_replan"
export REPLAN_URL="http://localhost:5000/replan"
export MEMORY_BASE_URL="http://localhost:5000"
export JUDGE_URL="http://localhost:8000/v1"
export JUDGE_PROMPT_TYPE="default"
export MAX_REFLECTION_TIMES=1
export MAX_LLM_CALL_PER_RUN=30
export TEST_CACHE_DIR="${DATA_DIR}/image_search_cache/fvqa_train_cache, ${DATA_DIR}/image_search_cache/fvqa_test_cache"

# ======================== 跑 5 条样本 ========================
echo "开始调试推理 (5 条样本)..."
cd "${CODE_DIR}/Inference/APEx-noTTRL"

curl -s -X POST http://localhost:5000/clear_memory > /dev/null 2>&1 || true

python -u run_multi_react.py \
    --dataset "${DEBUG_DATASET_SMALL}" \
    --output "${DEBUG_OUTPUT}" \
    --max_workers 1 \
    --max_tokens 10000 \
    --model "${DEBUG_MODEL}" \
    --temperature 0 \
    --presence_penalty 1.1 \
    --top_p 1.0 \
    --roll_out_count 1 \
    --main_ports "8000"

# ======================== 检查产出 ========================
echo ""
echo "=========================================="
echo "  调试完成，检查产出"
echo "=========================================="

if [ -f "${DEBUG_OUTPUT}/iter1.jsonl" ]; then
    total=$(wc -l < "${DEBUG_OUTPUT}/iter1.jsonl")
    echo "  iter1.jsonl: ${total} 条记录 ✓"
    echo "  前2条预览:"
    head -n 2 "${DEBUG_OUTPUT}/iter1.jsonl" | python3 -c "
import sys, json
for line in sys.stdin:
    d = json.loads(line)
    print(f\"    Q: {d.get('question','')[:60]}...\")
    print(f\"    J: {d.get('judgement','N/A')}  P: {d.get('prediction','')[:60]}...\")
    print()
" 2>/dev/null || head -n 2 "${DEBUG_OUTPUT}/iter1.jsonl"
else
    echo "  [失败] iter1.jsonl 未生成"
fi

if [ -f "${DEBUG_OUTPUT}/memory.jsonl" ]; then
    echo "  memory.jsonl: 已生成 ✓"
else
    echo "  [警告] memory.jsonl 未生成"
fi

# ======================== 清理 ========================
echo ""
cleanup_services
echo "调试结束！"
