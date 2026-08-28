#!/bin/bash
# ============================================================
# Skill-Guided Adaptive TTRL — fvqa_test — 8 卡完整版
#
# 创新点：
#   1. Skill-Gated Update: 高置信度 skill 跳过 GRPO 更新（节省计算）
#   2. Skill-Alignment Reward: 在 reward 中加入 skill 对齐分数（防遗忘）
#   3. Writer Model 驱动 skill 演化（GRPO训练的 Qwen3-8B）
#
# 与原始 run_ttrl_nogt_fvqa.sh 完全隔离：
#   - Agent Server 运行在 port 5001（原始 5000）
#   - 使用 mmsearch_skill.yaml 配置
#   - Checkpoint 保存到 TTRL-nogt-skill/ 目录
#   - 共享 Judge/Search 服务
#
# GPU 分配:
#   GPU 0,1 → Judge (Qwen3-32B, TP=2, port 8002)
#   GPU 2   → Executor (Trained-Executor, port 8000)
#   GPU 3   → Writer (Qwen3-8B, port 8004) — skill 合成与演化
#   GPU 4-7 → TTRL Planner 训练 (4卡 FSDP + sglang rollout)
#
# 服务架构:
#   Agent Serve nogt-skill (port 5001) → memory/plan/replan + skill + writer
#   Search (port 8001) → E5+FAISS 检索
# ============================================================

set -e

# ======================== 路径配置 ========================
BASE_DIR="/path/to/workdir"
BASE_DIR1="/home/jiangquan.sr"
CODE_DIR="${BASE_DIR1}/MIA-private"
DATA_DIR="${BASE_DIR}/dataset/MIA/dataset"
CKPT_DIR="${BASE_DIR}/models/MIA/MIA-checkpoint"
# OTHER_DIR="${BASE_DIR}/MIA-other"
SEARCH_R1_DIR="${BASE_DIR}/dataset/MIA/local_search"
# E5_MODEL="${OTHER_DIR}/e5-base-v2"
# FAISS_INDEX_DIR="${OTHER_DIR}/faiss_index"
# FAISS_INDEX="${FAISS_INDEX_DIR}/e5_Flat.index"
# WIKI25_MERGED="${OTHER_DIR}/wiki25_merged/wiki25.jsonl"
QWEN3_PATH="${BASE_DIR}/models/Qwen/Qwen3-32B"
LOG_DIR="${BASE_DIR1}/MIA-private/logs"

EXECUTOR_MODEL="/path/to/models/model/Trained-Executor"
PLANNER_MODEL="/path/to/models/planner"
WRITER_MODEL="/path/to/models/writer"

TTRL_NOGT_DIR="${CODE_DIR}/TTRL/TTRL-nogt"

# Writer 产出的 skill_repo.json（若存在则 warm-start）
SKILL_REPO_PATH="${DATA_DIR}/Train/writer/skill_repo.json"

# 训练配置 (与原始隔离的命名和路径)
PROJECT_NAME="deepresearch"
EXPERIMENT_NAME="ttrl_nogt_fvqa_skill_test"
SAVE_CHECKPOINT_DIR="/home/jiangquan.sr/ttrl_output/TTRL-nogt-skill/verl_checkpoints"

# fvqa_test 数据
FVQA_JSONL="${DATA_DIR}/Test/image-text/fvqa_test.jsonl"
FVQA_PARQUET="${DATA_DIR}/TTRL/Explore/fvqa_test_converted.parquet"

mkdir -p "${LOG_DIR}" "${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}" ./logs

echo "=========================================="
echo "  Skill-Guided Adaptive TTRL — fvqa_test"
echo "  Planner:  ${PLANNER_MODEL}"
echo "  Writer:   ${WRITER_MODEL}"
echo "  创新: Skill-Gated Update + Skill-Alignment Reward + Writer Evolution"
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

# ======================== Skill 配置 ========================
export SKILL_GATE_ENABLED="${SKILL_GATE_ENABLED:-1}"
export SKILL_HARD_GATE_ENABLED="${SKILL_HARD_GATE_ENABLED:-0}"
export SKILL_GATE_THRESHOLD="${SKILL_GATE_THRESHOLD:-0.98}"   # Diagnostic gate threshold
export SKILL_GATE_MIN_EVIDENCE="${SKILL_GATE_MIN_EVIDENCE:-20}"
export SKILL_GATE_REQUIRE_PROCEDURE="${SKILL_GATE_REQUIRE_PROCEDURE:-1}"
export SKILL_LAMBDA="0.1"           # Skill alignment reward 基础权重 λ
export SKILL_REWARD_SOURCE="${SKILL_REWARD_SOURCE:-auto}"  # auto: use GT accuracy when available, otherwise no-GT judge

# ======================== 辅助函数 ========================
wait_for_service() {
    local url=$1
    local name=$2
    local max_wait=${3:-60000}
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
    pkill -9 -f "agent_sever_ttrl_nogt_skill" 2>/dev/null || true
    pkill -9 -f "python.*retrieval_server.py" 2>/dev/null || true
    sleep 5
}

check_service() {
    curl -s --max-time 5 "$1" > /dev/null 2>&1
}

# ======================== Pre-check: 逐个检查服务状态 ========================
echo "========== [Pre-check] 检查依赖服务是否已就绪 =========="
NEED_JUDGE=false
NEED_EXECUTOR=false
NEED_WRITER=false
NEED_SEARCH=false
NEED_AGENT=false

if check_service "http://localhost:8002/v1/models"; then
    echo "  ✓ Judge (8002) 已就绪"
else
    echo "  ✗ Judge (8002) 未就绪，将启动"
    NEED_JUDGE=true
fi

if check_service "http://localhost:8000/v1/models"; then
    echo "  ✓ Executor (8000) 已就绪"
else
    echo "  ✗ Executor (8000) 未就绪，将启动"
    NEED_EXECUTOR=true
fi

if check_service "http://localhost:8004/v1/models"; then
    echo "  ✓ Writer (8004) 已就绪"
else
    echo "  ✗ Writer (8004) 未就绪，将启动"
    NEED_WRITER=true
fi

if check_service "http://localhost:8001/retrieve"; then
    echo "  ✓ Search (8001) 已就绪"
else
    echo "  ✗ Search (8001) 未就绪，将启动"
    NEED_SEARCH=true
fi

if check_service "http://localhost:5001/memory"; then
    echo "  ✓ Agent (5001) 已就绪"
else
    echo "  ✗ Agent (5001) 未就绪，将启动"
    NEED_AGENT=true
fi

# 判断是否全部就绪
if [ "$NEED_JUDGE" = "false" ] && [ "$NEED_EXECUTOR" = "false" ] && [ "$NEED_WRITER" = "false" ] && [ "$NEED_SEARCH" = "false" ] && [ "$NEED_AGENT" = "false" ]; then
    SERVICES_READY=true
    echo "  → 所有服务已就绪，跳过启动步骤"
else
    SERVICES_READY=false
    echo "  → 将仅启动缺失的服务（在线服务保持不动）"
fi

# ======================== Step 0: GPU 状态 ========================
if [ "$SERVICES_READY" = "false" ]; then
    echo "========== [Step 0] GPU 状态 =========="
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader 2>/dev/null || true
fi

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

if [ "$SERVICES_READY" = "false" ]; then

# ======================== Step 2: 按需启动 Judge ========================
if [ "$NEED_JUDGE" = "true" ]; then
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
        > "${LOG_DIR}/judge_skill_fvqa.log" 2>&1 &
    JUDGE_PID=$!
    echo "Judge PID: ${JUDGE_PID}"
else
    echo "========== [Step 2] Judge (8002) 已在线，跳过 =========="
fi

# ======================== Step 3: 按需启动 Executor ========================
if [ "$NEED_EXECUTOR" = "true" ]; then
    echo "========== [Step 3] 启动 Executor GPU 2 :8000 =========="
    CUDA_VISIBLE_DEVICES=2 \
    nohup vllm serve "${EXECUTOR_MODEL}" \
        --served-model-name "qwen" \
        --gpu-memory-utilization 0.85 \
        --enforce-eager \
        --host 0.0.0.0 \
        --port 8000 \
        --trust-remote-code \
        > "${LOG_DIR}/executor_skill_fvqa.log" 2>&1 &
    EXECUTOR_PID=$!
    echo "Executor PID: ${EXECUTOR_PID}"
else
    echo "========== [Step 3] Executor (8000) 已在线，跳过 =========="
fi

# ======================== Step 4: 按需启动 Writer ========================
if [ "$NEED_WRITER" = "true" ]; then
    echo "========== [Step 4] 启动 Writer (Qwen3-8B) GPU 3 :8004 =========="
    CUDA_VISIBLE_DEVICES=3 \
    nohup vllm serve "${WRITER_MODEL}" \
        --served-model-name "writer" \
        --gpu-memory-utilization 0.85 \
        --enforce-eager \
        --host 0.0.0.0 \
        --port 8004 \
        --trust-remote-code \
        --max-model-len 8192 \
        > "${LOG_DIR}/writer_skill_fvqa.log" 2>&1 &
    WRITER_PID=$!
    echo "Writer PID: ${WRITER_PID}"
else
    echo "========== [Step 4] Writer (8004) 已在线，跳过 =========="
fi

# ======================== Step 5: 按需启动 Search ========================
if [ "$NEED_SEARCH" = "true" ]; then
    echo "========== [Step 5] 启动 Search :8001 =========="
    cd "${SEARCH_R1_DIR}"
    nohup bash run.sh > "${LOG_DIR}/search_skill_fvqa.log" 2>&1 &
    SEARCH_PID=$!
    cd "${CODE_DIR}"
    echo "Search PID: ${SEARCH_PID}"
else
    echo "========== [Step 5] Search (8001) 已在线，跳过 =========="
fi

# ======================== 等待新启动的模型服务就绪 ========================
[ "$NEED_JUDGE" = "true" ] && wait_for_service "http://localhost:8002/v1/models" "Judge (8002)" 600
[ "$NEED_EXECUTOR" = "true" ] && wait_for_service "http://localhost:8000/v1/models" "Executor (8000)" 600
[ "$NEED_WRITER" = "true" ] && wait_for_service "http://localhost:8004/v1/models" "Writer (8004)" 600
[ "$NEED_SEARCH" = "true" ] && wait_for_service "http://localhost:8001/retrieve" "Search (8001)" 900

# ======================== Step 6: 按需启动 Agent Serve ========================
if [ "$NEED_AGENT" = "true" ]; then
    echo "========== [Step 6] 启动 Agent Serve nogt-skill :5001 (Skill-Guided + Writer) =========="
    export AGENT_URL="http://localhost:8000/v1"
    export SERVICE_URL="http://localhost:8001/"
    export MEMORY_URL="http://localhost:8002/v1"
    export WRITER_URL="http://localhost:8004/v1"
    export TEST_CACHE_DIR="/tmp/rrsun/MIA/image_search_cache/fvqa_train_cache,/tmp/rrsun/MIA/image_search_cache/fvqa_test_cache"
    export MAX_LLM_CALL_PER_RUN=20
    export TTRL_SAVE="${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}/memory_save"
    export PARQUET_PATH="${FVQA_PARQUET}"
    export IMAGE_JSONL_PATH="${FVQA_JSONL}"
    export SKILL_REPO_PATH="${SKILL_REPO_PATH}"
    export BERT_MODEL_PATH="${BASE_DIR}/models/MIA/sup-simcse-bert-base-uncased"
    mkdir -p "${TTRL_SAVE}"

    # Determine skill warm-start argument
    SKILL_ARG=""
    if [ -f "${SKILL_REPO_PATH}" ]; then
        SKILL_ARG="--skill_path ${SKILL_REPO_PATH}"
        echo "  Warm-start: 加载已有 skill_repo.json"
    else
        echo "  Cold-start: 无已有 skill，从空白开始"
    fi

    cd "${CODE_DIR}/Serve/MIA-TTRL"
    nohup python agent_sever_ttrl_nogt_skill.py \
        --model_name qwen \
        --port 5001 \
        --host 0.0.0.0 \
        --writer_url "http://localhost:8004/v1" \
        --writer_model writer \
        ${SKILL_ARG} \
        > "${LOG_DIR}/agent_serve_skill_fvqa.log" 2>&1 &
    AGENT_PID=$!
    cd "${CODE_DIR}"
    echo "Agent Serve nogt-skill PID: ${AGENT_PID}"

    wait_for_service "http://localhost:5001/memory" "Agent Serve nogt-skill (5001)" 300
else
    echo "========== [Step 6] Agent (5001) 已在线，跳过 =========="
fi

# ======================== Agent 环境变量（无论是否新启动都需要设置） ========================
export AGENT_URL="http://localhost:8000/v1"
export SERVICE_URL="http://localhost:8001/"
export MEMORY_URL="http://localhost:8002/v1"
export WRITER_URL="http://localhost:8004/v1"
export TEST_CACHE_DIR="/tmp/rrsun/MIA/image_search_cache/fvqa_train_cache,/tmp/rrsun/MIA/image_search_cache/fvqa_test_cache"
export MAX_LLM_CALL_PER_RUN=20
export TTRL_SAVE="${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}/memory_save"
export PARQUET_PATH="${FVQA_PARQUET}"
export IMAGE_JSONL_PATH="${FVQA_JSONL}"
export SKILL_REPO_PATH="${SKILL_REPO_PATH}"
export BERT_MODEL_PATH="${BASE_DIR}/models/MIA/sup-simcse-bert-base-uncased"
mkdir -p "${TTRL_SAVE}"

echo ""
echo "=========================================="
echo "  所有依赖服务已就绪！(Skill-Guided + Writer 版本)"
echo "  Judge (8002):    $([ "$NEED_JUDGE" = "true" ] && echo "新启动 PID=${JUDGE_PID}" || echo "复用已有")"
echo "  Executor (8000): $([ "$NEED_EXECUTOR" = "true" ] && echo "新启动 PID=${EXECUTOR_PID}" || echo "复用已有")"
echo "  Writer (8004):   $([ "$NEED_WRITER" = "true" ] && echo "新启动 PID=${WRITER_PID}" || echo "复用已有")"
echo "  Search (8001):   $([ "$NEED_SEARCH" = "true" ] && echo "新启动 PID=${SEARCH_PID}" || echo "复用已有")"
echo "  Agent (5001):    $([ "$NEED_AGENT" = "true" ] && echo "新启动 PID=${AGENT_PID}" || echo "复用已有")"
echo "  Gate: threshold=${SKILL_GATE_THRESHOLD}, lambda=${SKILL_LAMBDA}"
echo "=========================================="
echo ""

else
    echo ""
    echo "=========================================="
    echo "  复用已就绪服务（Judge/Executor/Writer/Search/Agent）"
    echo "  Gate: threshold=${SKILL_GATE_THRESHOLD}, lambda=${SKILL_LAMBDA}"
    echo "=========================================="
    echo ""
fi

# ======================== Step 7: 启动 Skill-Guided TTRL 训练 ========================
echo "========== [Step 7] 启动 Skill-Guided TTRL Planner 训练 GPU 4-7 =========="

# 训练进程连接 Agent Serve 的环境变量 (port 5001)
export JUDGE_URL="http://localhost:8002/v1"
export MEMORY_URL="http://localhost:5001/memory"
export PLAN_URL="http://localhost:5001/plan"
export REPLAN_URL="http://localhost:5001/replan"
export MEMORY_BANK_SAVE_URL="http://localhost:5001/memory_bank_save"
export BATCH_EVALUATE_URL="http://localhost:5001/batch_evaluate"
export TTRL_SKIP_BATCH_EVALUATE="${TTRL_SKIP_BATCH_EVALUATE:-1}"
export CONSOLIDATE_MEMORIES_URL="http://localhost:5001/consolidate_memories"
export SAVE_MEMORIES_URL="http://localhost:5001/save_memory"
echo "  Batch evaluate: TTRL_SKIP_BATCH_EVALUATE=${TTRL_SKIP_BATCH_EVALUATE}"

# Skill-specific URLs
export SKILL_MATCH_URL="http://localhost:5001/skill_match"
export SKILL_UPDATE_URL="http://localhost:5001/skill_update"
export SKILL_CONSOLIDATE_URL="http://localhost:5001/skill_consolidate"

cd "${TTRL_NOGT_DIR}"

# 清理残留的 Ray 状态，防止连接到过期的 GCS 地址
ray stop --force 2>/dev/null || true
rm -rf /tmp/ray/session_latest 2>/dev/null || true

CUDA_VISIBLE_DEVICES=4,5,6,7 \
RAY_ADDRESS="local" \
VLLM_USE_V1=1 \
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    --config-path="${TTRL_NOGT_DIR}/local_search/configs" \
    --config-name='mmsearch_skill' \
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
    actor_rollout_ref.rollout.name=vllm \
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
echo "  Skill-Guided TTRL 训练完成！"
echo "  Checkpoint: ${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}"
echo "  日志: ${LOG_DIR}/${EXPERIMENT_NAME}.log"
echo "=========================================="

# 保存 memory 和 skill repository
curl -s -X POST http://localhost:5001/save_memory > /dev/null 2>&1 || true
echo "已保存 memory 和 skill repository 到 ${TTRL_SAVE}"

# 打印最终 skill 统计
echo ""
echo "Skill Repository 最终统计:"
curl -s http://localhost:5001/skill_stats 2>/dev/null | python3 -m json.tool 2>/dev/null || true

# ======================== 推理输出统计与正确率 ========================
INFERENCE_OUTPUT="${TTRL_SAVE}/inference_outputs.jsonl"
echo ""
echo "=========================================="
echo "  推理输出分析"
echo "=========================================="
if [ -f "${INFERENCE_OUTPUT}" ]; then
    echo "  推理输出文件: ${INFERENCE_OUTPUT}"
    python3 -c "
import json, sys

results = []
with open('${INFERENCE_OUTPUT}', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            results.append(json.loads(line))

total = len(results)
if total == 0:
    print('  (无推理输出记录)')
    sys.exit(0)

def metric_judgement(r):
    # Prefer benchmark/GT judgement when present; fall back to the no-GT judge.
    return r.get('judgement_gt') or r.get('judgement_nogt')

correct = sum(1 for r in results if metric_judgement(r) == 'correct')
incorrect = sum(1 for r in results if metric_judgement(r) == 'incorrect')
gated = sum(1 for r in results if r.get('skill_gated') or r.get('judgement_nogt') == 'gated')
hard_gated = sum(1 for r in results if r.get('hard_gated') or r.get('judgement_nogt') == 'gated')
invalid = sum(1 for r in results if r.get('judgement_nogt') == 'invalid_format')
nogt_correct = sum(1 for r in results if r.get('judgement_nogt') == 'correct')
nogt_incorrect = sum(1 for r in results if r.get('judgement_nogt') == 'incorrect')
gt_available = sum(1 for r in results if r.get('judgement_gt') in ('correct', 'incorrect'))

print(f'  总样本数:     {total}')
print(f'  正确 (correct):  {correct}')
print(f'  错误 (incorrect): {incorrect}')
print(f'  Skill-Gate 命中: {gated}')
print(f'  Hard-Gated:      {hard_gated}')
print(f'  格式无效:        {invalid}')
print()

evaluated = correct + incorrect
if evaluated > 0:
    acc = correct / evaluated * 100
    print(f'  预测正确率 (优先 GT):    {correct}/{evaluated} = {acc:.2f}%')

nogt_evaluated = nogt_correct + nogt_incorrect
if nogt_evaluated > 0:
    nogt_acc = nogt_correct / nogt_evaluated * 100
    print(f'  No-GT judge 正确率:     {nogt_correct}/{nogt_evaluated} = {nogt_acc:.2f}%')
print(f'  GT judge 可用样本:      {gt_available}/{total}')

avg_reward = sum(r.get('reward', 0.0) for r in results) / total
print(f'  平均 reward:     {avg_reward:.4f}')

# 按 category 统计
print()
print('  === 按 category 分类统计 ===')
cats = {}
for r in results:
    c = r.get('category', 'unknown')
    if c not in cats:
        cats[c] = {'total': 0, 'correct': 0, 'incorrect': 0, 'gated': 0}
    cats[c]['total'] += 1
    j = metric_judgement(r)
    if j == 'correct':
        cats[c]['correct'] += 1
    elif j == 'incorrect':
        cats[c]['incorrect'] += 1
    if r.get('skill_gated') or r.get('judgement_nogt') == 'gated':
        cats[c]['gated'] += 1
for c, s in sorted(cats.items()):
    ev = s['correct'] + s['incorrect']
    acc_str = f'{s[\"correct\"]}/{ev} = {s[\"correct\"]/ev*100:.1f}%' if ev > 0 else 'N/A'
    print(f'    {c:12s}: total={s[\"total\"]:3d}  correct={s[\"correct\"]:3d}  incorrect={s[\"incorrect\"]:3d}  gated={s[\"gated\"]:3d}  acc={acc_str}')
"
else
    echo "  未找到推理输出文件: ${INFERENCE_OUTPUT}"
fi

if [ "$SERVICES_READY" = "false" ]; then
    echo "清理本次启动的服务..."
    [ "$NEED_JUDGE" = "true" ] && [ -n "${JUDGE_PID:-}" ] && kill -9 ${JUDGE_PID} 2>/dev/null || true
    [ "$NEED_EXECUTOR" = "true" ] && [ -n "${EXECUTOR_PID:-}" ] && kill -9 ${EXECUTOR_PID} 2>/dev/null || true
    [ "$NEED_WRITER" = "true" ] && [ -n "${WRITER_PID:-}" ] && kill -9 ${WRITER_PID} 2>/dev/null || true
    [ "$NEED_SEARCH" = "true" ] && [ -n "${SEARCH_PID:-}" ] && kill -9 ${SEARCH_PID} 2>/dev/null || true
    [ "$NEED_AGENT" = "true" ] && [ -n "${AGENT_PID:-}" ] && kill -9 ${AGENT_PID} 2>/dev/null || true
else
    echo "复用模式：保留已存在的服务，不执行清理"
fi
echo "全部完成！"
