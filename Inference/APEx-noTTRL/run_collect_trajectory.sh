#!/bin/bash
# ============================================================
# Stage 1→2 衔接：收集 Executor 执行轨迹
# 用途：固定 Planner (Qwen3-32B) + 训练好的 Executor (Stage 1)
#       在训练集上推理，收集轨迹供 Writer 训练使用
# 产出：iter1.jsonl + memory.jsonl
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
TRAIN_DATASET="${DATA_DIR}/Train/Executor/fvqa_train.jsonl"
OUTPUT_DIR="${RESULT_DIR}/writer_trajectory"

mkdir -p "${LOG_DIR}" "${RESULT_DIR}" "${OUTPUT_DIR}"

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

# ======================== Step 0: 环境准备 ========================
echo "========== [Step 0] 环境准备 =========="

cleanup_services

pip install flask soundfile scikit-learn fastapi uvicorn \
    qwen-agent openai json5 python-dotenv aiohttp \
    pyserini faiss-gpu -q 2>/dev/null || true

if [ ! -d "${SEARCH_R1_DIR}" ]; then
    echo "克隆 Search-R1 检索服务..."
    cd "${BASE_DIR}"
    git clone https://github.com/PeterGriffinJin/Search-R1.git
    cd -
fi

if [ ! -d "${E5_MODEL}" ]; then
    echo "[错误] E5 模型不存在: ${E5_MODEL}"
    exit 1
fi

if [ ! -f "${WIKI25_MERGED}" ]; then
    echo "合并 wiki25 分片文件..."
    mkdir -p "$(dirname ${WIKI25_MERGED})"
    cat ${OTHER_DIR}/wiki25/wiki25_part_* > "${WIKI25_MERGED}"
    echo "合并完成: $(du -sh ${WIKI25_MERGED} | cut -f1)"
fi

if [ ! -f "${FAISS_INDEX}" ]; then
    echo "构建 FAISS 索引..."
    mkdir -p "${FAISS_INDEX_DIR}"
    cd "${SEARCH_R1_DIR}"
    python search_r1/search/index_builder.py \
        --retrieval_method e5 \
        --model_path "${E5_MODEL}" \
        --corpus_path "${WIKI25_MERGED}" \
        --save_dir "${FAISS_INDEX_DIR}" \
        --use_fp16 \
        --max_length 256 \
        --batch_size 512 \
        --pooling_method mean \
        --faiss_type Flat \
        --save_embedding
    cd "${CODE_DIR}"
    echo "FAISS 索引构建完成: ${FAISS_INDEX}"
else
    echo "FAISS 索引已存在，跳过构建"
fi

if [ ! -f "${TRAIN_DATASET}" ]; then
    echo "[错误] 训练集不存在: ${TRAIN_DATASET}"
    exit 1
fi

echo "环境准备完成"

# ======================== Step 1: 启动 Judger & Planner (Qwen3-32B) ========================
echo "========== [Step 1] 启动 Judger & Planner (Qwen3-32B) :8002 =========="
export VLLM_USE_FLASHINFER_SAMPLER=0
CUDA_VISIBLE_DEVICES=0,1 \
nohup vllm serve "${QWEN3_PATH}" \
    --tensor-parallel-size 2 \
    --served-model-name "qwen" \
    --gpu-memory-utilization 0.8 \
    --host 0.0.0.0 \
    --port 8002 \
    --trust-remote-code \
    > "${LOG_DIR}/judger.log" 2>&1 &
JUDGER_PID=$!
echo "Judger & Planner PID: ${JUDGER_PID}"

# ======================== Step 2: 启动搜索服务 (Search-R1) ========================
echo "========== [Step 2] 启动搜索服务 (Search-R1 + E5) :8003 =========="
cd "${SEARCH_R1_DIR}"
sed -i 's/port=[0-9]\+/port=8003/g' search_r1/search/retrieval_server.py 2>/dev/null || true
nohup python search_r1/search/retrieval_server.py \
    --index_path "${FAISS_INDEX}" \
    --corpus_path "${WIKI25_MERGED}" \
    --topk 3 \
    --retriever_name e5 \
    --retriever_model "${E5_MODEL}" \
    > "${LOG_DIR}/search.log" 2>&1 &
SEARCH_PID=$!
cd "${CODE_DIR}"
echo "Search PID: ${SEARCH_PID}"

# ======================== 等待 Qwen3-32B 就绪 ========================
wait_for_service "http://localhost:8002/v1/models" "Judger & Planner (8002)" 600

# ======================== Step 3: 启动 Memory Service ========================
echo "========== [Step 3] 启动 Memory Service :5000 =========="
export MEMORY_URL="http://localhost:8002/v1"
export PLAN_URL="http://localhost:8002/v1"
cd "${CODE_DIR}/Memory-Serve"
nohup python memory_serve.py \
    --model_name qwen \
    --port 5000 \
    --host 0.0.0.0 \
    > "${LOG_DIR}/memory.log" 2>&1 &
MEMORY_PID=$!
echo "Memory PID: ${MEMORY_PID}"

# ======================== Step 4: 启动 Executor ========================
echo "========== [Step 4] 启动 Executor :8000 =========="
CUDA_VISIBLE_DEVICES=2,3 \
nohup vllm serve "${EXECUTOR_MODEL}" \
    --tensor-parallel-size 2 \
    --served-model-name "qwen" \
    --gpu-memory-utilization 0.8 \
    --enforce-eager \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    > "${LOG_DIR}/executor.log" 2>&1 &
EXECUTOR_PID=$!
echo "Executor PID: ${EXECUTOR_PID}"

# ======================== 等待所有服务就绪 ========================
wait_for_service "http://localhost:5000/hallo" "Memory Service (5000)" 120
wait_for_service "http://localhost:8000/v1/models" "Executor (8000)" 600
wait_for_service "http://localhost:8003/retrieve" "Search (8003)" 600

echo ""
echo "=========================================="
echo "  所有服务已就绪！"
echo "  Judger & Planner: PID=${JUDGER_PID}  port=8002"
echo "  Search:           PID=${SEARCH_PID}   port=8003"
echo "  Memory:           PID=${MEMORY_PID}   port=5000"
echo "  Executor:         PID=${EXECUTOR_PID} port=8000"
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

# ======================== 收集轨迹 ========================
echo ""
echo "=========================================="
echo "  开始收集 Executor 执行轨迹"
echo "  数据集: ${TRAIN_DATASET}"
echo "  输出:   ${OUTPUT_DIR}"
echo "=========================================="

cd "${CODE_DIR}/Inference/APEx-noTTRL"

curl -s -X POST http://localhost:5000/clear_memory > /dev/null 2>&1 || true

python -u run_multi_react.py \
    --dataset "${TRAIN_DATASET}" \
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
echo "  轨迹收集完成，统计结果"
echo "=========================================="

RESULT_FILE="${OUTPUT_DIR}/iter1.jsonl"
MEMORY_FILE="${OUTPUT_DIR}/memory.jsonl"

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
    echo "  轨迹文件: ${RESULT_FILE}"
else
    echo "  [警告] iter1.jsonl 未生成"
fi

if [ -f "${MEMORY_FILE}" ]; then
    echo "  记忆快照: ${MEMORY_FILE}"
else
    echo "  [警告] memory.jsonl 未生成"
fi

echo ""
echo "=========================================="
echo "  下一步: 用 build_writer_data.py 构建 Writer 训练数据"
echo "  方式A (从 memory.jsonl):"
echo "    cd ${CODE_DIR}/Writer-Train/Train"
echo "    python writer_skill/build_writer_data.py \\"
echo "        --input_memory ${MEMORY_FILE} \\"
echo "        --output_dir ${DATA_DIR}/Train/writer/"
echo ""
echo "  方式B (从 iter1.jsonl, 需分类服务):"
echo "    cd ${CODE_DIR}/Writer-Train/Train"
echo "    python writer_skill/build_writer_data.py \\"
echo "        --input_jsonl ${RESULT_FILE} \\"
echo "        --output_dir ${DATA_DIR}/Train/writer/ \\"
echo "        --classify_url http://127.0.0.1:8002/v1"
echo "=========================================="

# ======================== 清理服务 ========================
echo ""
echo "清理所有服务..."
cleanup_services
echo "全部完成！"
