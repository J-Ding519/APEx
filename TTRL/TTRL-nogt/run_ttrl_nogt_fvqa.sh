#!/bin/bash
# ============================================================
# TTRL-nogt 评测 fvqa_test — 8 卡完整版
# 原始 TTRL pipeline：无 skill，有 memory（从空白积累），无 ground-truth
#
# GPU 分配:
#   GPU 0,1 → Judge (Qwen3-32B, TP=2, port 8002)
#   GPU 2   → Executor 1 (Trained-Executor, port 8000)
#   GPU 3   → Executor 2 (Trained-Executor, port 8001)
#   GPU 4-7 → TTRL-nogt Planner 训练 (4卡 FSDP + sglang rollout)
#
# 服务架构:
#   Agent Serve nogt (port 5000) → memory/plan/replan，无 skill
#   Search (port 8003) → E5+FAISS 检索
#   无 GT Reward Server（nogt 版用 judge_nogt.py 内部评估）
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

EXECUTOR_MODEL="${CKPT_DIR}/Trained-Executor"
PLANNER_MODEL="/path/to/checkpoints/planner_n4/verl_checkpoints/mia_planner/planner_skill_grpo_n4/global_step_60/actor/huggingface"

TTRL_NOGT_DIR="${CODE_DIR}/TTRL/TTRL-nogt"

# 训练配置
PROJECT_NAME="deepresearch"
EXPERIMENT_NAME="ttrl_nogt_fvqa_test"
SAVE_CHECKPOINT_DIR="${CKPT_DIR}/TTRL-nogt/verl_checkpoints"

# fvqa_test 数据
FVQA_JSONL="${DATA_DIR}/Test/image-text/fvqa_test.jsonl"
FVQA_PARQUET="${DATA_DIR}/TTRL/Explore/fvqa_test_converted.parquet"

mkdir -p "${LOG_DIR}" "${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}" ./logs

echo "=========================================="
echo "  TTRL-nogt 评测 fvqa_test"
echo "  Planner:  ${PLANNER_MODEL}"
echo "  无 Skill / 有 Memory（空白积累）"
echo "  Dataset:  ${FVQA_PARQUET}"
echo "  Output:   ${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}"
echo "=========================================="

# ======================== 环境变量 ========================
export WANDB_MODE="offline"
export MODEL_MAX_LEN=32786
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_SHM_DISABLE=1
export NCCL_NET=Socket
export NCCL_DEBUG=WARN

# verl config 使用的环境变量
export TTRL_NOGT_DIR="${TTRL_NOGT_DIR}"

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
    pkill -9 -f "agent_sever_ttrl_nogt" 2>/dev/null || true
    pkill -9 -f "retrieval_server.py" 2>/dev/null || true
    sleep 5
}

# ======================== Step 0: 清理环境 ========================
echo "========== [Step 0] 清理环境 =========="
cleanup_services

echo "GPU 状态:"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader 2>/dev/null || true

# ======================== Step 1: 数据准备 ========================
echo "========== [Step 1] 数据准备 =========="
if [ ! -f "${FVQA_PARQUET}" ]; then
    echo "  转换 fvqa_test.jsonl → parquet ..."
    python3 -c "
import pandas as pd
df = pd.read_json('${FVQA_JSONL}', lines=True)
df.to_parquet('${FVQA_PARQUET}', index=False)
print(f'  转换完成: {len(df)} 条样本 → ${FVQA_PARQUET}')
"
else
    echo "  已存在: ${FVQA_PARQUET}"
fi

# ======================== Step 2: 启动 Judge (Qwen3-32B) ========================
echo "========== [Step 2] 启动 Judge (Qwen3-32B) GPU 0,1 :8002 =========="
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
    > "${LOG_DIR}/judge_ttrl_nogt_fvqa.log" 2>&1 &
JUDGE_PID=$!
echo "Judge PID: ${JUDGE_PID}"

# ======================== Step 3: 启动 Executor ========================
echo "========== [Step 3] 启动 Executor GPU 2 :8000, GPU 3 :8001 =========="
CUDA_VISIBLE_DEVICES=2 \
nohup vllm serve "${EXECUTOR_MODEL}" \
    --served-model-name "qwen" \
    --gpu-memory-utilization 0.85 \
    --enforce-eager \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    > "${LOG_DIR}/executor1_ttrl_nogt_fvqa.log" 2>&1 &
EXECUTOR1_PID=$!
echo "Executor 1 PID: ${EXECUTOR1_PID}"

CUDA_VISIBLE_DEVICES=3 \
nohup vllm serve "${EXECUTOR_MODEL}" \
    --served-model-name "qwen" \
    --gpu-memory-utilization 0.85 \
    --enforce-eager \
    --host 0.0.0.0 \
    --port 8001 \
    --trust-remote-code \
    > "${LOG_DIR}/executor2_ttrl_nogt_fvqa.log" 2>&1 &
EXECUTOR2_PID=$!
echo "Executor 2 PID: ${EXECUTOR2_PID}"

# ======================== Step 4: 启动 Search 服务 ========================
echo "========== [Step 4] 启动 Search :8003 =========="
cd "${SEARCH_R1_DIR}"
sed -i 's/port=[0-9]\+/port=8003/g' search_r1/search/retrieval_server.py 2>/dev/null || true
nohup python search_r1/search/retrieval_server.py \
    --index_path "${FAISS_INDEX}" \
    --corpus_path "${WIKI25_MERGED}" \
    --topk 3 \
    --retriever_name e5 \
    --retriever_model "${E5_MODEL}" \
    > "${LOG_DIR}/search_ttrl_nogt_fvqa.log" 2>&1 &
SEARCH_PID=$!
cd "${CODE_DIR}"
echo "Search PID: ${SEARCH_PID}"

# ======================== 等待模型服务就绪 ========================
wait_for_service "http://localhost:8002/v1/models" "Judge (8002)" 600
wait_for_service "http://localhost:8000/v1/models" "Executor 1 (8000)" 600
wait_for_service "http://localhost:8001/v1/models" "Executor 2 (8001)" 600

# ======================== Step 5: 启动 Agent Serve nogt ========================
echo "========== [Step 5] 启动 Agent Serve nogt :5000 (无 skill) =========="
export AGENT_URL="http://localhost:8000/v1,http://localhost:8001/v1"
export SERVICE_URL="http://localhost:8003/"
export MEMORY_URL="http://localhost:8002/v1"
export TEST_CACHE_DIR="${DATA_DIR}/image_search_cache/fvqa_train_cache,${DATA_DIR}/image_search_cache/fvqa_test_cache"
export MAX_LLM_CALL_PER_RUN=20
export TTRL_SAVE="${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}/memory_save"
mkdir -p "${TTRL_SAVE}"

cd "${CODE_DIR}/Serve/MIA-TTRL"
nohup python agent_sever_ttrl_nogt.py \
    --model_name qwen \
    --port 5000 \
    --host 0.0.0.0 \
    > "${LOG_DIR}/agent_serve_nogt_fvqa.log" 2>&1 &
AGENT_PID=$!
cd "${CODE_DIR}"
echo "Agent Serve nogt PID: ${AGENT_PID}"

# ======================== 等待所有服务就绪 ========================
wait_for_service "http://localhost:8003/retrieve" "Search (8003)" 900
wait_for_service "http://localhost:5000/memory" "Agent Serve nogt (5000)" 300

echo ""
echo "=========================================="
echo "  所有依赖服务已就绪！"
echo "  Judge:       PID=${JUDGE_PID}       GPU 0,1  port=8002"
echo "  Executor 1:  PID=${EXECUTOR1_PID}   GPU 2    port=8000"
echo "  Executor 2:  PID=${EXECUTOR2_PID}   GPU 3    port=8001"
echo "  Search:      PID=${SEARCH_PID}               port=8003"
echo "  Agent Serve: PID=${AGENT_PID}                port=5000 (nogt, 无 skill)"
echo "=========================================="
echo ""

# ======================== Step 6: 启动 TTRL-nogt 训练 ========================
echo "========== [Step 6] 启动 TTRL-nogt Planner 训练 GPU 4-7 =========="

# 训练进程连接 Agent Serve 的环境变量
export JUDGE_URL="http://localhost:8002/v1"
export MEMORY_URL="http://localhost:5000/memory"
export PLAN_URL="http://localhost:5000/plan"
export REPLAN_URL="http://localhost:5000/replan"
export MEMORY_BANK_SAVE_URL="http://localhost:5000/memory_bank_save"
export BATCH_EVALUATE_URL="http://localhost:5000/batch_evaluate"
export CONSOLIDATE_MEMORIES_URL="http://localhost:5000/consolidate_memories"
export SAVE_MEMORIES_URL="http://localhost:5000/save_memory"

cd "${TTRL_NOGT_DIR}"

CUDA_VISIBLE_DEVICES=4,5,6,7 \
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    --config-path="${TTRL_NOGT_DIR}/local_search/configs" \
    --config-name='mmsearch' \
    data.train_files="${FVQA_PARQUET}" \
    data.val_files=["${FVQA_PARQUET}"] \
    data.train_batch_size=8 \
    data.max_prompt_length=24576 \
    data.max_response_length=8192 \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    data.shuffle=False \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.model.path="${PLANNER_MODEL}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.optim.lr=2e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','hf_model','optimizer','extra'] \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.max_num_batched_tokens=65536 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=10 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=10 \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=2 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=8192 \
    trainer.critic_warmup=0 \
    trainer.rollout_data_dir="${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}/rollout_saved" \
    trainer.validation_data_dir="${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}/val_saved" \
    trainer.logger=['console','tensorboard'] \
    trainer.val_before_train=False \
    trainer.val_only=False \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=2000 \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.default_local_dir="${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}" \
    +trainer.tensorboard_dir="${SAVE_CHECKPOINT_DIR}/logs/tensorboard" \
    +trainer.rl_logging_board_dir="${SAVE_CHECKPOINT_DIR}/logs/rl_logging_board" \
    trainer.total_epochs=1 \
    2>&1 | tee "${LOG_DIR}/${EXPERIMENT_NAME}.log"

# ======================== 训练结束 ========================
echo ""
echo "=========================================="
echo "  TTRL-nogt 训练完成！"
echo "  Checkpoint: ${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}"
echo "  日志: ${LOG_DIR}/${EXPERIMENT_NAME}.log"
echo "=========================================="

curl -s -X POST http://localhost:5000/save_memory > /dev/null 2>&1 || true
echo "已保存 memory 到 ${TTRL_SAVE}"

cleanup_services
echo "全部完成！"
