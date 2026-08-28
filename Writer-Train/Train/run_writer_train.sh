#!/bin/bash
# ============================================================
# Writer GRPO 训练启动脚本
# 用途：启动 Judge 服务 + 运行 Writer GRPO 训练
# GPU 分配：GPU 0,1 → Judge (Qwen3-32B)
#           GPU 2-7 → Writer Training (6卡)
# ============================================================

set -eo pipefail

export WANDB_API_KEY=""
export WANDB_MODE="online"

# ======================== 路径配置 ========================
BASE_DIR="/path/to/workdir"
CODE_DIR="${BASE_DIR}/MIA-private"
CKPT_DIR="${BASE_DIR}/MIA-checkpoint"
QWEN3_PATH="/path/to/Qwen3-32B"
LOG_DIR="${BASE_DIR}/logs"

mkdir -p "${LOG_DIR}" "${CODE_DIR}/Writer-Train/Train/logs"

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
    echo "清理 Judge 服务..."
    pkill -f "vllm serve.*port 8001" 2>/dev/null || true
    sleep 3
}

# ======================== Step 0: 清理残留 ========================
echo "========== [Step 0] 清理残留服务 =========="
cleanup_services

# ======================== Step 1: 启动 Judge 服务 ========================
echo "========== [Step 1] 启动 Judge (Qwen3-32B) :8001 =========="
export VLLM_USE_FLASHINFER_SAMPLER=0
CUDA_VISIBLE_DEVICES=0,1 \
nohup vllm serve "${QWEN3_PATH}" \
    --tensor-parallel-size 2 \
    --served-model-name "qwen" \
    --gpu-memory-utilization 0.8 \
    --host 0.0.0.0 \
    --port 8001 \
    --trust-remote-code \
    > "${LOG_DIR}/writer_judge.log" 2>&1 &
JUDGE_PID=$!
echo "Judge PID: ${JUDGE_PID}"

# ======================== 等待 Judge 就绪 ========================
wait_for_service "http://localhost:8001/v1/models" "Judge (8001)" 600

echo ""
echo "=========================================="
echo "  Judge 服务已就绪！"
echo "  Judge: PID=${JUDGE_PID}  port=8001"
echo "=========================================="
echo ""

# ======================== Step 2: 运行 Writer GRPO 训练 ========================
echo "========== [Step 2] 启动 Writer GRPO 训练 (GPU 2-7) =========="

export JUDGE_URL="http://localhost:8001/v1"
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7

cd "${CODE_DIR}/Writer-Train/Train"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    --config-path="${CODE_DIR}/Writer-Train/Train/writer_skill/configs" \
    --config-name='writer_grpo' \
    data.train_files="${BASE_DIR}/MIA-data/Train/writer/writer_train.parquet" \
    data.val_files=[${BASE_DIR}/MIA-data/Train/writer/writer_val.parquet] \
    data.train_batch_size=48 \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.model.path="/path/to/Qwen3-8B" \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.use_fused_kernels=False \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.multi_turn.enable=False \
    trainer.critic_warmup=0 \
    trainer.rollout_data_dir="${CKPT_DIR}/writer/verl_checkpoints/mia_writer/writer_skill_grpo/rollout_saved" \
    trainer.validation_data_dir="${CKPT_DIR}/writer/verl_checkpoints/mia_writer/writer_skill_grpo/val_saved" \
    trainer.logger=['console','wandb','tensorboard'] \
    trainer.val_before_train=False \
    trainer.val_only=False \
    trainer.n_gpus_per_node=6 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=20 \
    trainer.project_name="mia_writer" \
    trainer.experiment_name="writer_skill_grpo" \
    trainer.default_local_dir="${CKPT_DIR}/writer/verl_checkpoints/mia_writer/writer_skill_grpo" \
    +trainer.tensorboard_dir="${CKPT_DIR}/writer/verl_checkpoints/logs/tensorboard" \
    +trainer.rl_logging_board_dir="${CKPT_DIR}/writer/verl_checkpoints/logs/rl_logging_board" \
    trainer.total_epochs=8 2>&1 | tee ./logs/writer_skill_grpo.log

# ======================== 训练结束，清理 ========================
echo ""
echo "=========================================="
echo "  Writer GRPO 训练完成！"
echo "  Checkpoint: ${CKPT_DIR}/writer/verl_checkpoints/mia_writer/writer_skill_grpo"
echo "=========================================="

cleanup_services
echo "全部完成！"
