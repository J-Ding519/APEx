#!/bin/bash
# ============================================================
# 构建 Planner 训练数据（注入 skill_context）
# 一键脚本：启动分类服务 → 等待就绪 → 构建数据 → 关闭服务
# GPU: 2 卡 (TP=2)
# ============================================================

set -e

# ======================== 路径配置 ========================
BASE_DIR="/path/to/workdir"
CODE_DIR="${BASE_DIR}/MIA-private"
DATA_DIR="${BASE_DIR}/MIA-data"
LOG_DIR="${BASE_DIR}/logs"
QWEN3_PATH="/path/to/Qwen3-32B"

INPUT_PARQUET="${DATA_DIR}/Train/Planner/fvqa_matpo_train_planner.parquet"
SKILL_REPO="${DATA_DIR}/Train/writer/skill_repo.json"
OUTPUT_PARQUET="${DATA_DIR}/Train/Planner/planner_with_skill.parquet"

mkdir -p "${LOG_DIR}"

# ======================== 辅助函数 ========================
wait_for_service() {
    local url=$1
    local name=$2
    local max_wait=${3:-600}
    local waited=0
    echo "[等待] ${name} 启动中..."
    while ! curl -s "${url}" > /dev/null 2>&1; do
        sleep 10
        waited=$((waited + 10))
        if [ ${waited} -ge ${max_wait} ]; then
            echo "[超时] ${name} 在 ${max_wait}s 内未就绪"
            exit 1
        fi
        echo "  ... 已等待 ${waited}s"
    done
    echo "[就绪] ${name} 已启动 (${waited}s)"
}

# ======================== Step 1: 启动分类服务 ========================
echo "========== 启动 Qwen3-32B 分类服务 (GPU 0,1 :8002) =========="
pkill -9 -f "vllm serve" 2>/dev/null || true
sleep 3

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
    > "${LOG_DIR}/classify_service.log" 2>&1 &
VLLM_PID=$!
echo "vLLM PID: ${VLLM_PID}"

wait_for_service "http://localhost:8002/v1/models" "Qwen3-32B (8002)" 600

# ======================== Step 2: 构建数据 ========================
echo "========== 构建 Planner 训练数据 =========="
echo "  输入: ${INPUT_PARQUET}"
echo "  Skill: ${SKILL_REPO}"
echo "  输出: ${OUTPUT_PARQUET}"

cd "${CODE_DIR}/Planner-Train/mem-plan"
python local_search/build_planner_data.py \
    --input_parquet "${INPUT_PARQUET}" \
    --skill_repo "${SKILL_REPO}" \
    --output_parquet "${OUTPUT_PARQUET}" \
    --classify_url http://localhost:8002/v1

# ======================== Step 3: 拆分 train/val ========================
echo "========== 拆分 Train/Val (90%/10%) =========="
TRAIN_PARQUET="${DATA_DIR}/Train/Planner/planner_with_skill_train.parquet"
VAL_PARQUET="${DATA_DIR}/Train/Planner/planner_with_skill_val.parquet"

python -c "
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_parquet('${OUTPUT_PARQUET}')
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['category'])
train_df.to_parquet('${TRAIN_PARQUET}', index=False)
val_df.to_parquet('${VAL_PARQUET}', index=False)
print(f'Train: {len(train_df)} samples')
print(f'Val:   {len(val_df)} samples')
print(f'Train skill coverage: {(train_df[\"skill_context\"] != \"No skill available for this question type.\").sum()}/{len(train_df)}')
print(f'Val skill coverage:   {(val_df[\"skill_context\"] != \"No skill available for this question type.\").sum()}/{len(val_df)}')
"

# ======================== Step 4: 清理 ========================
echo "========== 清理分类服务 =========="
kill ${VLLM_PID} 2>/dev/null || true
pkill -9 -f "vllm serve" 2>/dev/null || true

echo ""
echo "=========================================="
echo "  完成！"
echo "  训练集: ${TRAIN_PARQUET}"
echo "  验证集: ${VAL_PARQUET}"
echo "  下一步: bash local_search/run_mmsearch_grpo.sh"
echo "=========================================="
