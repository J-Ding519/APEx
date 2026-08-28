#!/bin/bash
# ============================================================
# Writer-Planner-Executor 完整推理 Pipeline
# 使用训练好的 Planner (n4, step=60) 替代 Qwen3-32B 做规划
# GPU 分配：
#   GPU 0,1 → Judger (Qwen3-32B, port 8002)
#   GPU 2   → Writer (Qwen3-8B, trained, port 8001)
#   GPU 3   → Executor (trained, port 8000)
#   GPU 4   → Planner (trained Qwen3-8B, step60, port 8004)
# ============================================================

set -e

# ======================== 路径配置 ========================
BASE_DIR="/path/to/workdir"
CODE_DIR="${BASE_DIR}/MIA-private"
DATA_DIR="${BASE_DIR}/MIA-data"
CKPT_DIR="${BASE_DIR}/MIA-checkpoint"
OTHER_DIR="${BASE_DIR}/MIA-other"
SEARCH_R1_DIR="${BASE_DIR}/Search-R1"
E5_MODEL="${OTHER_DIR}/e5-base-v2"
FAISS_INDEX_DIR="${OTHER_DIR}/faiss_index"
FAISS_INDEX="${FAISS_INDEX_DIR}/e5_Flat.index"
WIKI25_MERGED="${OTHER_DIR}/wiki25_merged/wiki25.jsonl"
QWEN3_PATH="/path/to/Qwen3-32B"
LOG_DIR="${BASE_DIR}/logs"
RESULT_DIR="${BASE_DIR}/results"

EXECUTOR_MODEL="${CKPT_DIR}/Trained-Executor"
WRITER_MODEL="${CKPT_DIR}/writer/verl_checkpoints/mia_writer/writer_skill_grpo/global_step_40/actor/huggingface"
PLANNER_MODEL="${CKPT_DIR}/planner_n4/verl_checkpoints/mia_planner/planner_skill_grpo_n4/global_step_60/actor/huggingface"
SKILL_REPO="${DATA_DIR}/Train/writer/skill_repo.json"

# 推理数据集（默认测试集）
TEST_DATASET="${1:-${DATA_DIR}/Train/Executor/fvqa_test.jsonl}"
OUTPUT_DIR="${RESULT_DIR}/writer_inference_planner_step60"

# Skill 进化参数
SKILL_EVOLVE_THRESHOLD=5

mkdir -p "${LOG_DIR}" "${RESULT_DIR}" "${OUTPUT_DIR}"

echo "=========================================="
echo "  Writer-Planner-Executor 推理 Pipeline"
echo "  Planner: trained (n4, step=60)"
echo "  Dataset: ${TEST_DATASET}"
echo "  Output:  ${OUTPUT_DIR}"
echo "  Skill Repo: ${SKILL_REPO}"
echo "  Writer: ${WRITER_MODEL}"
echo "  Planner: ${PLANNER_MODEL}"
echo "  Executor: ${EXECUTOR_MODEL}"
echo "=========================================="

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
    pkill -9 -f "vllm serve" 2>/dev/null || true
    pkill -9 -f "memory_serve.py" 2>/dev/null || true
    pkill -9 -f "retrieval_server.py" 2>/dev/null || true
    sleep 5
}

# ======================== Step 0: 环境准备 ========================
echo "========== [Step 0] 环境准备 =========="
cleanup_services

echo "GPU 状态:"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "nvidia-smi 不可用"

# NCCL 环境变量（解决多卡通信问题）
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_SHM_DISABLE=1
export NCCL_NET=Socket
export NCCL_DEBUG=WARN

# ======================== Step 1: 启动 Judger (Qwen3-32B) ========================
echo "========== [Step 1] 启动 Judger (Qwen3-32B) :8002 =========="
export VLLM_USE_FLASHINFER_SAMPLER=0
CUDA_VISIBLE_DEVICES=0,1 \
nohup vllm serve "${QWEN3_PATH}" \
    --tensor-parallel-size 2 \
    --served-model-name "qwen" \
    --gpu-memory-utilization 0.9 \
    --enforce-eager \
    --host 0.0.0.0 \
    --port 8002 \
    --trust-remote-code \
    > "${LOG_DIR}/judger_writer_inf.log" 2>&1 &
JUDGER_PID=$!
echo "Judger PID: ${JUDGER_PID}"

# ======================== Step 2: 启动 Planner (trained, step=60) ========================
echo "========== [Step 2] 启动 Planner (trained step=60) :8004 =========="
CUDA_VISIBLE_DEVICES=4 \
nohup vllm serve "${PLANNER_MODEL}" \
    --served-model-name "planner" \
    --gpu-memory-utilization 0.85 \
    --enforce-eager \
    --host 0.0.0.0 \
    --port 8004 \
    --trust-remote-code \
    --max-model-len 32768 \
    > "${LOG_DIR}/planner_inf.log" 2>&1 &
PLANNER_PID=$!
echo "Planner PID: ${PLANNER_PID}"

# ======================== Step 3: 启动 Writer (Qwen3-8B trained) ========================
echo "========== [Step 3] 启动 Writer :8001 =========="
CUDA_VISIBLE_DEVICES=2 \
nohup vllm serve "${WRITER_MODEL}" \
    --served-model-name "writer" \
    --gpu-memory-utilization 0.85 \
    --enforce-eager \
    --host 0.0.0.0 \
    --port 8001 \
    --trust-remote-code \
    --max-model-len 8192 \
    > "${LOG_DIR}/writer_inf.log" 2>&1 &
WRITER_PID=$!
echo "Writer PID: ${WRITER_PID}"

# ======================== Step 4: 启动 Executor ========================
echo "========== [Step 4] 启动 Executor :8000 =========="
CUDA_VISIBLE_DEVICES=3 \
nohup vllm serve "${EXECUTOR_MODEL}" \
    --served-model-name "qwen" \
    --gpu-memory-utilization 0.85 \
    --enforce-eager \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    > "${LOG_DIR}/executor_writer_inf.log" 2>&1 &
EXECUTOR_PID=$!
echo "Executor PID: ${EXECUTOR_PID}"

# ======================== Step 5: 启动搜索服务 (Search-R1) ========================
echo "========== [Step 5] 启动搜索服务 :8003 =========="
cd "${SEARCH_R1_DIR}"
sed -i 's/port=[0-9]\+/port=8003/g' search_r1/search/retrieval_server.py 2>/dev/null || true
nohup python search_r1/search/retrieval_server.py \
    --index_path "${FAISS_INDEX}" \
    --corpus_path "${WIKI25_MERGED}" \
    --topk 3 \
    --retriever_name e5 \
    --retriever_model "${E5_MODEL}" \
    > "${LOG_DIR}/search_writer_inf.log" 2>&1 &
SEARCH_PID=$!
cd "${CODE_DIR}"
echo "Search PID: ${SEARCH_PID}"

# ======================== 等待模型服务就绪 ========================
wait_for_service "http://localhost:8002/v1/models" "Judger (8002)" 600
wait_for_service "http://localhost:8004/v1/models" "Planner (8004)" 600
wait_for_service "http://localhost:8001/v1/models" "Writer (8001)" 600
wait_for_service "http://localhost:8000/v1/models" "Executor (8000)" 600

# ======================== Step 6: 启动 Memory Service (带 skill + writer) ========================
echo "========== [Step 6] 启动 Memory Service :5000 (skill-enhanced) =========="
export MEMORY_URL="http://localhost:8002/v1"
export PLAN_URL="http://localhost:8004/v1"
cd "${CODE_DIR}/Memory-Serve"

# 构建 Memory-Serve 启动参数
MEMORY_ARGS="--model_name qwen --port 5000 --host 0.0.0.0"
MEMORY_ARGS="${MEMORY_ARGS} --writer_url http://localhost:8001/v1 --writer_model writer"
MEMORY_ARGS="${MEMORY_ARGS} --skill_evolve_threshold ${SKILL_EVOLVE_THRESHOLD}"

# 如果有预生成的 skill_repo.json，加载作为初始 skill
if [ -f "${SKILL_REPO}" ]; then
    MEMORY_ARGS="${MEMORY_ARGS} --skill_repo ${SKILL_REPO}"
    echo "  加载初始 skill: ${SKILL_REPO}"
else
    echo "  无初始 skill，将在推理过程中从零积累"
fi

nohup python memory_serve.py ${MEMORY_ARGS} \
    > "${LOG_DIR}/memory_writer_inf.log" 2>&1 &
MEMORY_PID=$!
echo "Memory PID: ${MEMORY_PID}"

# ======================== 等待 Memory 和 Search 就绪 ========================
wait_for_service "http://localhost:5000/hallo" "Memory Service (5000)" 600
wait_for_service "http://localhost:8003/retrieve" "Search (8003)" 600

echo ""
echo "=========================================="
echo "  所有服务已就绪！"
echo "  Judger:   PID=${JUDGER_PID}   GPU 0,1  port=8002"
echo "  Planner:  PID=${PLANNER_PID}  GPU 4    port=8004 (step=60)"
echo "  Writer:   PID=${WRITER_PID}   GPU 2    port=8001"
echo "  Executor: PID=${EXECUTOR_PID} GPU 3    port=8000"
echo "  Search:   PID=${SEARCH_PID}            port=8003"
echo "  Memory:   PID=${MEMORY_PID}            port=5000"
echo "=========================================="
echo ""

# ======================== 推理环境变量 ========================
export SERVICE_URL="http://localhost:8003/"
export PLAN_URL="http://localhost:5000/plan"
export REPLAN_JUDGE_URL="http://localhost:5000/judge_replan"
export REPLAN_URL="http://localhost:5000/replan"
export MEMORY_BASE_URL="http://localhost:5000"
export JUDGE_URL="http://localhost:8002/v1"
export JUDGE_PROMPT_TYPE="default"
export MAX_REFLECTION_TIMES=1
export MAX_LLM_CALL_PER_RUN=30
export TEST_CACHE_DIR="${DATA_DIR}/image_search_cache/fvqa_train_cache, ${DATA_DIR}/image_search_cache/fvqa_test_cache"

MODEL_PATH="${EXECUTOR_MODEL}"

# ======================== 开始推理 ========================
echo ""
echo "=========================================="
echo "  开始 Writer-Planner-Executor 推理"
echo "  Planner: trained (n4, step=60)"
echo "  数据集: ${TEST_DATASET}"
echo "  输出:   ${OUTPUT_DIR}"
echo "  Skill 进化: 每 ${SKILL_EVOLVE_THRESHOLD} 条新记忆触发"
echo "=========================================="

cd "${CODE_DIR}/Inference/APEx-noTTRL"

curl -s -X POST http://localhost:5000/clear_memory > /dev/null 2>&1 || true

python -u run_multi_react.py \
    --dataset "${TEST_DATASET}" \
    --output "${OUTPUT_DIR}" \
    --max_workers 16 \
    --max_tokens 10000 \
    --model "${MODEL_PATH}" \
    --temperature 0 \
    --presence_penalty 1.1 \
    --top_p 1.0 \
    --roll_out_count 1 \
    --main_ports "8000"

# ======================== 统计结果 ========================
echo ""
echo "=========================================="
echo "  推理完成，统计结果 (Planner step=60)"
echo "=========================================="

RESULT_FILE="${OUTPUT_DIR}/iter1.jsonl"

if [ -f "${RESULT_FILE}" ]; then
    total=$(wc -l < "${RESULT_FILE}")
    correct=$(grep -c '"judgement": "correct"' "${RESULT_FILE}" 2>/dev/null || echo 0)
    incorrect=$(grep -c '"judgement": "incorrect"' "${RESULT_FILE}" 2>/dev/null || echo 0)
    failed=$(grep -c '"prediction": "\[Failed\]"' "${RESULT_FILE}" 2>/dev/null || echo 0)

    if [ ${total} -gt 0 ]; then
        accuracy=$(python3 -c "print(f'{${correct}/${total}*100:.2f}')")
    else
        accuracy="0.00"
    fi

    echo "  总样本: ${total}"
    echo "  正确:   ${correct} (${accuracy}%)"
    echo "  错误:   ${incorrect}"
    echo "  失败:   ${failed}"
    echo ""
    echo "  结果文件: ${RESULT_FILE}"
else
    echo "  [警告] iter1.jsonl 未生成"
fi

cleanup_services
echo "全部完成！"
