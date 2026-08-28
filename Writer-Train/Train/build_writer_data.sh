#!/bin/bash
# ============================================================
# Step 3: 构建 Writer 训练数据（方式 B：从 iter1.jsonl + LLM 分类）
# 前置：轨迹收集已完成，iter1.jsonl 已生成
# 产出：writer_train.parquet + writer_val.parquet
# ============================================================

set -e

# ======================== 路径配置 ========================
BASE_DIR="/path/to/workdir"
CODE_DIR="${BASE_DIR}/MIA-private"
DATA_DIR="${BASE_DIR}/MIA-data"
CKPT_DIR="${BASE_DIR}/MIA-checkpoint"
OTHER_DIR="${BASE_DIR}/MIA-other"
RESULT_DIR="${BASE_DIR}/results"
LOG_DIR="${BASE_DIR}/logs"

QWEN3_PATH="/path/to/Qwen3-32B"

INPUT_JSONL="${RESULT_DIR}/writer_trajectory/iter1.jsonl"
OUTPUT_DIR="${DATA_DIR}/Train/writer"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

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

# ======================== 前置检查 ========================
if [ ! -f "${INPUT_JSONL}" ]; then
    echo "[错误] iter1.jsonl 不存在: ${INPUT_JSONL}"
    exit 1
fi

echo "========== [Step 3] 构建 Writer 训练数据 =========="
echo "  输入: ${INPUT_JSONL}"
echo "  输出: ${OUTPUT_DIR}"
echo ""

# ======================== Step 1: 启动 Qwen3-32B 分类服务 ========================
echo "========== 启动 Qwen3-32B 分类服务 :8002 =========="
export VLLM_USE_FLASHINFER_SAMPLER=0
CUDA_VISIBLE_DEVICES=0,1 \
nohup vllm serve "${QWEN3_PATH}" \
    --tensor-parallel-size 2 \
    --served-model-name "qwen" \
    --gpu-memory-utilization 0.8 \
    --host 0.0.0.0 \
    --port 8002 \
    --trust-remote-code \
    > "${LOG_DIR}/classify_qwen3.log" 2>&1 &
CLASSIFY_PID=$!
echo "分类服务 PID: ${CLASSIFY_PID}"

wait_for_service "http://localhost:8002/v1/models" "Qwen3-32B 分类服务 (8002)" 600

# ======================== Step 2: 运行 build_writer_data.py ========================
echo ""
echo "========== 运行 build_writer_data.py =========="

cd "${CODE_DIR}/Writer-Train/Train"

python -u writer_skill/build_writer_data.py \
    --input_jsonl "${INPUT_JSONL}" \
    --output_dir "${OUTPUT_DIR}" \
    --classify_url http://127.0.0.1:8002/v1 \
    --classify_model qwen \
    --batch_min 5 \
    --batch_max 15 \
    --val_ratio 0.2 \
    --seed 42

# ======================== 结果统计 ========================
echo ""
echo "=========================================="
echo "  Writer 训练数据构建完成"
echo "=========================================="

if [ -f "${OUTPUT_DIR}/writer_train.parquet" ]; then
    echo "  训练集: ${OUTPUT_DIR}/writer_train.parquet"
    echo "  验证集: ${OUTPUT_DIR}/writer_val.parquet"
else
    echo "  [警告] parquet 文件未生成，请检查日志"
fi

echo ""
echo "=========================================="
echo "  下一步: 训练 Writer"
echo "    cd ${CODE_DIR}/Writer-Train/Train"
echo "    # 1. 修改 writer_skill/configs/writer_grpo.yaml 数据路径"
echo "    # 2. 修改 writer_skill/run_writer_grpo.sh 模型路径"
echo "    export JUDGE_URL=http://127.0.0.1:8002/v1"
echo "    bash writer_skill/run_writer_grpo.sh"
echo "=========================================="

# ======================== 清理服务 ========================
echo ""
echo "关闭分类服务..."
kill ${CLASSIFY_PID} 2>/dev/null || true
sleep 3
echo "完成！"
