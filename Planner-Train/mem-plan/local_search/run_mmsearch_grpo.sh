#!/bin/bash
# ============================================================
# Planner GRPO 训练 (Stage 3) — 8 卡完整版
# GPU 分配:
#   GPU 0,1 → Judge (Qwen3-32B, TP=2, port 8002)
#   GPU 2   → Executor (Trained-Executor, port 8000, for Agent Serve)
#   GPU 3   → (Search 服务用 CPU，不占 GPU)
#   GPU 4-7 → Planner 训练 (4卡 FSDP + sglang rollout)
#
# 执行流程:
#   本脚本会自动启动所有依赖服务，然后开始训练
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
QWEN3_8B_PATH="/path/to/Qwen3-8B"
LOG_DIR="${BASE_DIR}/logs"

EXECUTOR_MODEL="${CKPT_DIR}/Trained-Executor"

PROJECT_NAME="mia_planner"
EXPERIMENT_NAME="planner_skill_grpo"
SAVE_CHECKPOINT_DIR="${CKPT_DIR}/planner/verl_checkpoints"

# 训练数据（run_build_planner_data.sh 的输出，已拆分 train/val）
DATASET_TRAIN="${DATA_DIR}/Train/Planner/planner_with_skill_train.parquet"
DATASET_VAL="${DATA_DIR}/Train/Planner/planner_with_skill_val.parquet"

# 基座模型
REF_MODEL_PATH="${QWEN3_8B_PATH}"

mkdir -p "${LOG_DIR}" "${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}" ./logs

# ======================== 环境变量 ========================
export WANDB_API_KEY=""
export WANDB_MODE="online"
export MODEL_MAX_LEN=32786
export JUDGE_URL="http://localhost:8002/v1"
export PLAN_URL="http://localhost:5000/plan"
export REPLAN_URL="http://localhost:5000/replan"

# NCCL 环境变量
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_SHM_DISABLE=1
export NCCL_NET=Socket
export NCCL_DEBUG=WARN

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

cleanup_services() {
    echo "清理服务..."
    pkill -9 -f "vllm serve" 2>/dev/null || true
    pkill -9 -f "agent_serve.py" 2>/dev/null || true
    pkill -9 -f "retrieval_server.py" 2>/dev/null || true
    sleep 5
}

# ======================== Step 0: 清理环境 ========================
echo "========== [Step 0] 清理环境 =========="
cleanup_services

echo "GPU 状态:"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader 2>/dev/null || true

# ======================== Step 1: 启动 Judge (Qwen3-32B) ========================
echo "========== [Step 1] 启动 Judge (Qwen3-32B) GPU 0,1 :8002 =========="
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
    > "${LOG_DIR}/judge_planner_train.log" 2>&1 &
JUDGE_PID=$!
echo "Judge PID: ${JUDGE_PID}"

# ======================== Step 2: 启动 Executor (for Agent Serve) ========================
echo "========== [Step 2] 启动 Executor GPU 2 :8000, GPU 3 :8001 =========="
CUDA_VISIBLE_DEVICES=2 \
nohup vllm serve "${EXECUTOR_MODEL}" \
    --served-model-name "qwen" \
    --gpu-memory-utilization 0.85 \
    --enforce-eager \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    > "${LOG_DIR}/executor_planner_train.log" 2>&1 &
EXECUTOR_PID=$!
echo "Executor 1 PID: ${EXECUTOR_PID}"

CUDA_VISIBLE_DEVICES=3 \
nohup vllm serve "${EXECUTOR_MODEL}" \
    --served-model-name "qwen" \
    --gpu-memory-utilization 0.85 \
    --enforce-eager \
    --host 0.0.0.0 \
    --port 8001 \
    --trust-remote-code \
    > "${LOG_DIR}/executor2_planner_train.log" 2>&1 &
EXECUTOR2_PID=$!
echo "Executor 2 PID: ${EXECUTOR2_PID}"

# ======================== Step 3: 启动 Search 服务 ========================
echo "========== [Step 3] 启动 Search :8003 =========="
cd "${SEARCH_R1_DIR}"
sed -i 's/port=[0-9]\+/port=8003/g' search_r1/search/retrieval_server.py 2>/dev/null || true
nohup python search_r1/search/retrieval_server.py \
    --index_path "${FAISS_INDEX}" \
    --corpus_path "${WIKI25_MERGED}" \
    --topk 3 \
    --retriever_name e5 \
    --retriever_model "${E5_MODEL}" \
    > "${LOG_DIR}/search_planner_train.log" 2>&1 &
SEARCH_PID=$!
cd "${CODE_DIR}"
echo "Search PID: ${SEARCH_PID}"

# ======================== 等待模型服务就绪 ========================
wait_for_service "http://localhost:8002/v1/models" "Judge (8002)" 600
wait_for_service "http://localhost:8000/v1/models" "Executor 1 (8000)" 600
wait_for_service "http://localhost:8001/v1/models" "Executor 2 (8001)" 600

# ======================== Step 4: 启动 Agent Serve ========================
echo "========== [Step 4] 启动 Agent Serve :5000 =========="
export SERVICE_URL="http://localhost:8003/"
export AGENT_URL="http://localhost:8000/v1,http://localhost:8001/v1"
export TEST_CACHE_DIR="/path/to/data/image_search_cache/fvqa_train_cache, /path/to/data/image_search_cache/fvqa_test_cache"
export MAX_LLM_CALL_PER_RUN=20
cd "${CODE_DIR}/Serve/Train_Planner"
nohup gunicorn -w 4 --threads 16 -b 0.0.0.0:5000 --timeout 600 agent_serve:app \
    > "${LOG_DIR}/agent_serve_planner_train.log" 2>&1 &
AGENT_SERVE_PID=$!
cd "${CODE_DIR}"
echo "Agent Serve PID: ${AGENT_SERVE_PID}"

# Search 加载索引+语料需要较长时间，先等 Search 就绪
wait_for_service "http://localhost:8003/retrieve" "Search (8003)" 900
wait_for_service "http://localhost:5000/plan" "Agent Serve (5000)" 300

echo ""
echo "=========================================="
echo "  所有服务已就绪！"
echo "  Judge:       PID=${JUDGE_PID}       GPU 0,1  port=8002"
echo "  Executor 1:  PID=${EXECUTOR_PID}    GPU 2    port=8000"
echo "  Executor 2:  PID=${EXECUTOR2_PID}   GPU 3    port=8001"
echo "  Search:      PID=${SEARCH_PID}               port=8003"
echo "  Agent Serve: PID=${AGENT_SERVE_PID}          port=5000"
echo "  训练将使用:   GPU 4,5,6,7 (4卡)"
echo "=========================================="
echo ""

# ======================== Step 5: 启动 Planner 训练 ========================
echo "========== [Step 5] 启动 Planner GRPO 训练 =========="

cd "${CODE_DIR}/Planner-Train/mem-plan"

set -x
CUDA_VISIBLE_DEVICES=4,5,6,7 \
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    --config-path="${CODE_DIR}/Planner-Train/mem-plan/local_search/configs" \
    --config-name='mmsearch' \
    data.train_files=${DATASET_TRAIN} \
    data.val_files=[${DATASET_VAL}] \
    data.train_batch_size=48 \
    data.max_prompt_length=24576 \
    data.max_response_length=8192 \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.model.path=${REF_MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=48 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','hf_model','optimizer','extra'] \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
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
    trainer.rollout_data_dir=${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}/rollout_saved \
    trainer.validation_data_dir=${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}/val_saved \
    trainer.logger=['console','wandb','tensorboard'] \
    trainer.val_before_train=False \
    trainer.val_only=False \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=20 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME} \
    +trainer.tensorboard_dir=${SAVE_CHECKPOINT_DIR}/logs/tensorboard \
    +trainer.rl_logging_board_dir=${SAVE_CHECKPOINT_DIR}/logs/rl_logging_board \
    trainer.total_epochs=5 2>&1 | tee ${LOG_DIR}/${EXPERIMENT_NAME}.log

# ======================== 清理 ========================
echo "训练完成，清理服务..."
cleanup_services
echo "全部完成！"
