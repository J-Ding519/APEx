#!/bin/bash
# ============================================================
# 生成 skill_repo.json
# 用途：用训练好的 Writer 模型对每个 (category, modality) 的记忆
#       进行推理，生成结构化 skill，汇总为 skill_repo.json
# GPU 分配：GPU 0,1 → Writer 模型 (Qwen3-8B checkpoint)
# ============================================================

set -eo pipefail

# ======================== 路径配置 ========================
BASE_DIR="/path/to/workdir"
CODE_DIR="${BASE_DIR}/MIA-private"
CKPT_DIR="${BASE_DIR}/MIA-checkpoint"
RESULT_DIR="${BASE_DIR}/results"
LOG_DIR="${BASE_DIR}/logs"

# Writer checkpoint (step 40, val reward 最高)
WRITER_CKPT="${1:-${CKPT_DIR}/writer/verl_checkpoints/mia_writer/writer_skill_grpo/global_step_40/actor/huggingface}"

# 记忆数据来源（iter1.jsonl 用于轨迹收集阶段产出）
INPUT_JSONL="${RESULT_DIR}/writer_trajectory/iter1.jsonl"

# 输出
OUTPUT_FILE="${BASE_DIR}/MIA-data/Train/writer/skill_repo.json"

mkdir -p "${LOG_DIR}" "$(dirname ${OUTPUT_FILE})"

echo "=========================================="
echo "  生成 skill_repo.json"
echo "  Writer Checkpoint: ${WRITER_CKPT}"
echo "  Input: ${INPUT_JSONL}"
echo "  Output: ${OUTPUT_FILE}"
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
    echo "清理服务..."
    pkill -f "vllm serve.*port 8000" 2>/dev/null || true
    pkill -f "vllm serve.*port 8002" 2>/dev/null || true
    sleep 3
}

# ======================== Step 0: 清理残留 ========================
cleanup_services

# ======================== Step 1: 启动分类服务 (Qwen3-32B) ========================
echo "========== [Step 1] 启动 Qwen3-32B 分类服务 :8002 =========="
QWEN3_PATH="/path/to/Qwen3-32B"
export VLLM_USE_FLASHINFER_SAMPLER=0

CUDA_VISIBLE_DEVICES=0,1 \
nohup vllm serve "${QWEN3_PATH}" \
    --tensor-parallel-size 2 \
    --served-model-name "qwen" \
    --gpu-memory-utilization 0.8 \
    --host 0.0.0.0 \
    --port 8002 \
    --trust-remote-code \
    > "${LOG_DIR}/classify_qwen3_skill.log" 2>&1 &
CLASSIFY_PID=$!
echo "分类服务 PID: ${CLASSIFY_PID}"

wait_for_service "http://localhost:8002/v1/models" "Qwen3-32B 分类服务 (8002)" 600

# ======================== Step 2: 启动 Writer 模型服务 ========================
echo "========== [Step 2] 启动 Writer 模型 :8000 =========="

CUDA_VISIBLE_DEVICES=2,3 \
nohup vllm serve "${WRITER_CKPT}" \
    --served-model-name "writer" \
    --gpu-memory-utilization 0.85 \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --max-model-len 8192 \
    > "${LOG_DIR}/writer_skill_gen.log" 2>&1 &
WRITER_PID=$!
echo "Writer 服务 PID: ${WRITER_PID}"

wait_for_service "http://localhost:8000/v1/models" "Writer 模型 (8000)" 600

echo ""
echo "=========================================="
echo "  服务已就绪"
echo "  分类: PID=${CLASSIFY_PID} port=8002"
echo "  Writer: PID=${WRITER_PID} port=8000"
echo "=========================================="
echo ""

# ======================== Step 3: 运行 skill 生成 ========================
echo "========== [Step 3] 生成 skill_repo.json =========="

cd "${CODE_DIR}/Writer-Train/Train"

python -u writer_skill/generate_skill_repo.py \
    --input_jsonl "${INPUT_JSONL}" \
    --model_url http://localhost:8000/v1 \
    --model_name writer \
    --output "${OUTPUT_FILE}" \
    --batch_size 15 \
    --min_memories 3 \
    --max_tokens 2048 \
    --classify_url http://localhost:8002/v1 \
    --classify_model qwen \
    --verbose

# ======================== 完成 ========================
echo ""
echo "=========================================="
echo "  skill_repo.json 生成完成！"
echo "  输出: ${OUTPUT_FILE}"
echo ""
echo "  下一步: 构建 Planner 训练数据"
echo "    cd Planner-Train/mem-plan"
echo "    python local_search/build_planner_data.py \\"
echo "        --skill_repo ${OUTPUT_FILE} ..."
echo "=========================================="

cleanup_services
echo "全部完成！"
