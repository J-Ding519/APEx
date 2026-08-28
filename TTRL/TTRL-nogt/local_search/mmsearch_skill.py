"""
Skill-Guided Adaptive TTRL: mmsearch_skill.py

Implements two innovations over original mmsearch.py:
1. Skill Gate Diagnostics — high-confidence skills are marked in logs; optional
   hard-gating can still be enabled explicitly for ablation/protection runs
2. Skill-Alignment Reward — adds skill alignment score to reward function
   to provide knowledge-level regularization against catastrophic forgetting
"""
import io
import json
import logging
import os
import random
import re
import torch
import requests
from openai import OpenAI
from PIL import Image
import asyncio
import verl.utils.torch_functional as verl_F
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.model import compute_position_id_with_mask
import time
from local_search.prompt import *
from local_search.judge_nogt import evaluate_agent_without_gt

logger = logging.getLogger(__name__)

# ============================================================
# LLM client for judge (shared with original pipeline)
# ============================================================
openai_api_key = "EMPTY"
openai_api_base = os.getenv("JUDGE_URL")
model_name = "qwen"
_openai_client = None


def get_openai_client():
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
    return _openai_client


def truncate_by_whitespace_words(text: str, max_words: int = 1024) -> str:
    if not text.strip():
        return text
    words = text.split()
    if len(words) > max_words:
        truncated_words = words[:max_words]
        truncated_words.append("... (Omitted part of the results returned by the tool)")
    else:
        truncated_words = words[:max_words]
    return " ".join(truncated_words)


def get_format_workflow(question, temp_messages):
    trace = f"### Question: {question}\n"
    j = 1
    for message in temp_messages:
        role = message["role"]
        content = message["content"]
        if isinstance(content, str):
            text_content = content
        elif isinstance(content, list):
            text_parts = []
            for item in content:
                if item["type"] == "text":
                    text = item.get("text", "").strip()
                    text_parts.append(text)
            text_content = " ".join(text_parts)
        else:
            text_content = str(content)
        if role == "assistant":
            trace += f"### Round {j}:\n"
            trace += f"#### Agent Reasoning and Tool Call:\n{text_content}\n"
            j += 1
        else:
            text_content = truncate_by_whitespace_words(text_content)
            trace += f"#### Tool Call Return Results:\n{text_content}\n"
    return trace


class CustomRLHFDataset(RLHFDataset):
    def __getitem__(self, item):
        row_dict: dict = self.dataframe[item]
        question = row_dict[self.prompt_key][0]["content"]
        modality = row_dict["modality"]
        if "image_caption" in row_dict.keys():
            if row_dict["image_caption"]:
                image_caption = row_dict["image_caption"]
                if isinstance(image_caption, list):
                    image_caption = image_caption[0] if image_caption else ""
                else:
                    image_caption = image_caption if image_caption is not None else ""
            else:
                image_caption = ""
        else:
            image_caption = ""
        plan_prompt_format = PLAN_PROMPT
        if modality == "text-only":
            row_dict[self.prompt_key] = [
                {
                    "role": "system",
                    "content": SYSTEM_PLAN_PROMPT,
                },
            ]
            plan_prompt_format = PLAN_PROMPT
        else:
            row_dict[self.prompt_key] = [
                {
                    "role": "system",
                    "content": SYSTEM_PLAN_PROMPT,
                },
            ]
            plan_prompt_format = PLAN_PROMPT_IMG
        messages = self._build_messages(row_dict)

        model_inputs = {}

        if self.processor is not None:
            raw_prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=False, tokenize=False
            )
            model_inputs = self.processor(text=[raw_prompt], return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")
        else:
            raw_prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=False, tokenize=False
            )
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(
                    f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}."
                )

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages
        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = dict()
        if "category" in row_dict and row_dict["category"] is not None:
            row_dict["extra_info"]["category"] = row_dict["category"]
        if "level" in row_dict and row_dict["level"] is not None:
            row_dict["extra_info"]["level"] = row_dict["level"]
        if "data_source" in row_dict and row_dict["data_source"] is not None:
            row_dict["extra_info"]["data_source"] = row_dict["data_source"]
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt

        replan_prompt = REPLAN_PROMPT.format(question=question)
        row_dict["extra_info"]["slow_replan"] = ""
        row_dict["extra_info"]["question"] = question
        row_dict["extra_info"]["image_caption"] = image_caption
        row_dict["extra_info"]["plan_prompt_format"] = plan_prompt_format
        row_dict["extra_info"]["modality"] = row_dict["modality"]
        row_dict["extra_info"]["data_id"] = row_dict["data_id"]
        row_dict["extra_info"]["replan_prompt"] = replan_prompt
        row_dict["extra_info"]["messages"] = []
        row_dict["agent_name"] = "multi_turn_agent"
        return row_dict


def normalize_text(text):
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.strip())


def extract_after_think(text):
    last_think_index_close = text.rfind("</think>")
    last_think_index_open = text.rfind("<think>")
    last_think_index = max(last_think_index_open, last_think_index_close)
    if last_think_index == -1:
        return text.replace("<|im_end|>", "").strip()
    start_index = last_think_index + len("</think>")
    result = text[start_index:]
    result = result.replace("<|im_end|>", "").strip()
    return result


def extract_answer(solution_str: str) -> str:
    predict_no_think = extract_after_think(solution_str)
    answer_match = re.search(r"<answer>(.*?)</answer>", predict_no_think, re.DOTALL)
    if answer_match:
        answer_text = answer_match.group(1).strip()
    else:
        tool_response_match = re.search(
            r"<tool_call>\s*assistant\s*\n(.*?)$", predict_no_think, re.DOTALL | re.MULTILINE
        )
        if tool_response_match:
            answer_text = tool_response_match.group(1).strip()
        else:
            if "</think>" in solution_str:
                remaining_content = predict_no_think
                remaining_content = re.sub(
                    r"<tool_call>.*?<tool_call>", "", remaining_content, flags=re.DOTALL
                )
                remaining_content = re.sub(
                    r"\b(?:user|assistant)\b", "", remaining_content, flags=re.IGNORECASE
                )
                answer_text = remaining_content.strip()
            else:
                answer_text = solution_str.strip()
    answer_text = answer_text.replace("<|im_end|>", "").strip()
    if not answer_text:
        answer_text = solution_str.strip()
    answer_text = normalize_text(answer_text)
    return answer_text


def judge_answer2(question_text, ans, ground_truth) -> bool:
    if not ans.strip():
        return False
    user_prompt = JUDGE_PROMPT.format(
        question=question_text,
        correct_answer=ground_truth,
        response=ans,
    )
    try:
        chat_response = get_openai_client().chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": user_prompt}],
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            temperature=0.0,
        )
        response = chat_response.choices[0].message.content.strip()
        response = response.replace("\\n", "\n").replace("\\r", "\r")
        cleaned = extract_after_think(response).strip()
        if cleaned == "A":
            return True
        if cleaned in ("B", "C"):
            return False
        return False
    except Exception:
        print("Judger Connect Error")
        pt = ans.lower().strip()
        gt = str(ground_truth).lower().strip()
        return gt in pt

# ============================================================
# Skill-related URLs
# ============================================================
memory_bank_save_url = os.getenv("MEMORY_BANK_SAVE_URL")
skill_match_url = os.getenv("SKILL_MATCH_URL")
skill_update_url = os.getenv("SKILL_UPDATE_URL")

# ============================================================
# Skill-Gated Update config
# ============================================================
GATE_ENABLED = os.getenv("SKILL_GATE_ENABLED", "1").lower() not in ("0", "false", "no")
HARD_GATE_ENABLED = os.getenv("SKILL_HARD_GATE_ENABLED", "0").lower() in ("1", "true", "yes")
GATE_THRESHOLD = float(os.getenv("SKILL_GATE_THRESHOLD", "0.98"))
GATE_MIN_EVIDENCE = int(os.getenv("SKILL_GATE_MIN_EVIDENCE", "20"))
GATE_REQUIRE_PROCEDURE = os.getenv("SKILL_GATE_REQUIRE_PROCEDURE", "1").lower() not in ("0", "false", "no")
BASE_LAMBDA = float(os.getenv("SKILL_LAMBDA", "0.1"))
REWARD_SOURCE = os.getenv("SKILL_REWARD_SOURCE", "auto").lower()

# ============================================================
# Inference output logging
# ============================================================
INFERENCE_OUTPUT_DIR = os.getenv("TTRL_SAVE", ".")
INFERENCE_OUTPUT_FILE = os.path.join(INFERENCE_OUTPUT_DIR, "inference_outputs.jsonl")
_inference_log_lock = None


def get_inference_log_lock():
    global _inference_log_lock
    if _inference_log_lock is None:
        _inference_log_lock = __import__("threading").Lock()
    return _inference_log_lock


def _format_executor_trace(question: str, temp_messages: list) -> str:
    trace = f"### Question: {question}\n"
    j = 1
    for msg in temp_messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, list):
            parts = [item.get("text", "") for item in content if item.get("type") == "text"]
            text = " ".join(parts)
            image_count = sum(1 for item in content if item.get("type") in ("image", "image_url"))
            if image_count:
                text = f"{text}\n[attached_images={image_count}]"
        else:
            text = str(content)
        if role == "assistant":
            trace += f"### Round {j}:\n#### Agent Reasoning and Tool Call:\n{text}\n"
            j += 1
        else:
            trace += f"#### Tool Call Return Results:\n{text}\n"
    return trace


def log_inference_output(record: dict):
    os.makedirs(os.path.dirname(INFERENCE_OUTPUT_FILE), exist_ok=True)
    with get_inference_log_lock():
        try:
            with open(INFERENCE_OUTPUT_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as e:
            print(f"[WARNING] log_inference_output failed: {e}", flush=True)

# ============================================================
# Skill-Alignment Reward prompt
# ============================================================
SKILL_ALIGNMENT_PROMPT = """Evaluate whether the following plan follows the recommended skill procedure and avoids known pitfalls.

**Skill Procedure (recommended steps):**
{procedure}

**Known Pitfalls to Avoid:**
{pitfalls}

**Generated Plan:**
{plan}

Score the alignment from 0.0 to 1.0:
- 1.0: Plan follows all procedure steps and avoids all pitfalls
- 0.7: Plan follows most steps with minor deviations
- 0.5: Plan partially follows the procedure or misses some pitfalls
- 0.3: Plan only loosely relates to the skill guidance
- 0.0: Plan completely ignores the skill guidance

Output ONLY a single decimal number between 0.0 and 1.0, nothing else."""


def _format_procedure(procedure) -> str:
    """Format Writer's structured procedure list into readable text."""
    if not procedure:
        return "(no procedure available)"
    if isinstance(procedure, str):
        return procedure
    lines = []
    for step in procedure:
        if isinstance(step, dict):
            n = step.get("step", "")
            action = step.get("action", "")
            purpose = step.get("purpose", "")
            lines.append(f"{n}. {action} — {purpose}" if purpose else f"{n}. {action}")
        else:
            lines.append(str(step))
    return "\n".join(lines)


def _format_pitfalls(pitfalls) -> str:
    """Format Writer's pitfalls list into readable text."""
    if not pitfalls:
        return "(no pitfalls recorded)"
    if isinstance(pitfalls, str):
        return pitfalls
    return "\n".join(f"- {p}" for p in pitfalls)


def compute_skill_alignment(plan: str, skill_data: dict) -> float:
    procedure = skill_data.get("procedure", [])
    pitfalls = skill_data.get("pitfalls", [])
    if not procedure and not pitfalls:
        return 0.0
    prompt = SKILL_ALIGNMENT_PROMPT.format(
        procedure=_format_procedure(procedure),
        pitfalls=_format_pitfalls(pitfalls),
        plan=plan,
    )
    try:
        response = get_openai_client().chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=16,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        text = response.choices[0].message.content.strip()
        text = re.sub(r'[^0-9.]', '', text)
        score = float(text)
        return max(0.0, min(1.0, score))
    except Exception as e:
        logger.warning(f"Skill alignment scoring failed: {e}")
        return 0.0


def adaptive_lambda(confidence: float) -> float:
    return BASE_LAMBDA * confidence


def call_skill_match(category: str, modality: str) -> dict:
    if not skill_match_url:
        return {"confidence": 0.0, "skill": None}
    try:
        resp = requests.post(
            skill_match_url,
            json={"category": category, "modality": modality},
            timeout=10,
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning(f"Skill match call failed: {e}")
        return {"confidence": 0.0, "skill": None}


def call_skill_update(category: str, modality: str, is_correct: bool):
    if not skill_update_url:
        return
    try:
        requests.post(
            skill_update_url,
            json={"category": category, "modality": modality, "is_correct": is_correct},
            timeout=10,
            headers={"Content-Type": "application/json"},
        )
    except Exception as e:
        logger.warning(f"Skill update call failed: {e}")


def compute_format_score(messages: list) -> float:
    format_score = 0.0
    do_replan = extract_after_think(messages[2]).lower()
    if do_replan in ["yes", "no"]:
        format_score = 1.0

    indices_to_check = [0, 2]
    if len(messages) == 6:
        indices_to_check.append(4)
    for idx in indices_to_check:
        msg = messages[idx]
        if msg.count("<think>") != msg.count("</think>"):
            format_score = 0.0
            break
    return format_score


def compute_ground_truth_judgement(question_text: str, model_answer: str, ground_truth: str):
    if not str(ground_truth).strip():
        return None, None
    try:
        is_correct = judge_answer2(question_text, model_answer, ground_truth)
    except Exception as e:
        logger.warning(f"Ground-truth judging failed: {e}")
        return None, None
    judgement = "correct" if is_correct else "incorrect"
    return judgement, 1.0 if is_correct else 0.0


def should_gate_update(confidence: float, skill: dict) -> bool:
    if not GATE_ENABLED or confidence < GATE_THRESHOLD or not skill:
        return False
    evidence_count = int(skill.get("evidence_count", 0) or 0)
    if evidence_count < GATE_MIN_EVIDENCE:
        return False
    if GATE_REQUIRE_PROCEDURE and not skill.get("procedure"):
        return False
    return True


def select_reward_accuracy(accuracy_score: float, accuracy_score_gt):
    if REWARD_SOURCE == "gt":
        if accuracy_score_gt is not None:
            return accuracy_score_gt, "gt"
        return accuracy_score, "nogt_fallback"
    if REWARD_SOURCE == "nogt":
        return accuracy_score, "nogt"
    # auto: use benchmark accuracy when available, otherwise no-GT judge accuracy.
    if accuracy_score_gt is not None:
        return accuracy_score_gt, "gt"
    return accuracy_score, "nogt"


# ============================================================
# Main reward function
# ============================================================
def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info=None) -> float:
    messages = extra_info.get("messages", [])
    question_text = extra_info.get("question", "")
    if len(messages) not in (3, 6):
        log_inference_output({
            "data_id": extra_info.get("data_id", ""),
            "question": question_text,
            "ground_truth": ground_truth,
            "model_answer": "",
            "plan": "",
            "judgement_nogt": "invalid_format",
            "reward": 0.0,
            "category": extra_info.get("category", "others"),
            "modality": extra_info.get("modality", "text-only"),
            "executor_trace": "",
        })
        return 0.0

    # ---- Step 1: Skill Matching ----
    category = extra_info.get("category", "others")
    modality = extra_info.get("modality", "text-only")
    skill_result = call_skill_match(category, modality)
    confidence = skill_result.get("confidence", 0.0)
    skill = skill_result.get("skill")

    # ---- Step 2: Extract basic info for memory saving ----
    model_answer = extract_answer(messages[5]) if len(messages) == 6 else extract_answer(messages[1])
    data_id = extra_info.get("data_id", "")
    plan = extract_after_think(messages[0]).strip()
    image_caption = extra_info.get("image_caption", "").strip()
    used_memory_indices = extra_info.get("used_memory_indices", [])
    temp_messages = extra_info.get("temp_messages", [])
    executor_trace = _format_executor_trace(question_text, temp_messages)
    format_score = compute_format_score(messages)
    format_workflow = get_format_workflow(question_text, temp_messages)
    judgement_data = evaluate_agent_without_gt(
        question=question_text, trajectory=format_workflow, final_output=model_answer
    )
    judgement = judgement_data["is_correct"]
    accuracy_score = 1.0 if judgement == "correct" else 0.0
    judgement_gt, accuracy_score_gt = compute_ground_truth_judgement(
        question_text, model_answer, ground_truth
    )

    skill_gated = should_gate_update(confidence, skill)
    reward_accuracy_score, actual_reward_source = select_reward_accuracy(
        accuracy_score, accuracy_score_gt
    )

    # ---- Step 3: Optional Skill-Gated Update (disabled by default) ----
    if skill_gated and HARD_GATE_ENABLED:
        # High confidence: return fixed reward for all rollouts of this sample.
        # Since all n rollouts get the same reward, GRPO advantage = 0 → no parameter update.
        # This mode is for explicit ablations/protection runs, not the default TTRL path.
        logger.info(
            f"[SKILL-HARD-GATE] Skipping TTRL update for {data_id} "
            f"(category={category}, modality={modality}, confidence={confidence:.3f}, "
            f"evidence={skill.get('evidence_count', 0)}, judgement={judgement})"
        )
        gated_judgement_data = {
            **judgement_data,
            "skill_gated": True,
            "hard_gated": True,
            "skill_confidence": confidence,
            "skill_evidence_count": skill.get("evidence_count", 0),
        }
        data = {
            "data_id": data_id,
            "plan": plan,
            "question": question_text,
            "model_answer": model_answer,
            "ground_truth": ground_truth,
            "image_caption": image_caption,
            "used_memory_indices": used_memory_indices,
            "temp_messages": temp_messages,
            "judgement_nogt": judgement,
            "judgement_data": gated_judgement_data,
        }
        get_user_response_from_url(memory_bank_save_url, data)
        call_skill_update(category, modality, is_correct=(reward_accuracy_score >= 1.0))

        log_inference_output({
            "data_id": data_id,
            "question": question_text,
            "ground_truth": ground_truth,
            "model_answer": model_answer,
            "plan": plan,
            "judgement_nogt": judgement,
            "judgement_data": gated_judgement_data,
            "judgement_gt": judgement_gt,
            "reward": confidence,
            "format_score": format_score,
            "accuracy_score": accuracy_score,
            "accuracy_score_gt": accuracy_score_gt,
            "reward_accuracy_score": reward_accuracy_score,
            "reward_source": actual_reward_source,
            "category": category,
            "modality": modality,
            "skill_gated": True,
            "hard_gated": True,
            "skill_confidence": confidence,
            "skill_evidence_count": skill.get("evidence_count", 0),
            "executor_trace": executor_trace,
        })
        return confidence

    if skill_gated:
        logger.info(
            f"[SKILL-GATE-DIAGNOSTIC] Using accuracy reward for {data_id} "
            f"(category={category}, modality={modality}, confidence={confidence:.3f}, "
            f"evidence={skill.get('evidence_count', 0) if skill else 0}, judgement={judgement})"
        )

    # ---- Step 4: Accuracy-based reward for normal TTRL update ----
    original_reward = 0.9 * reward_accuracy_score + 0.1 * format_score

    # ---- Step 5: Skill-Alignment Reward ----
    if skill and (skill.get("procedure") or skill.get("pitfalls")):
        alignment_score = compute_skill_alignment(plan, skill)
        lambda_skill = adaptive_lambda(confidence)
        final_reward = (1.0 - lambda_skill) * original_reward + lambda_skill * alignment_score
    else:
        final_reward = original_reward

    # ---- Step 6: Update Skill Stats & Save to Memory Bank ----
    is_correct = (reward_accuracy_score >= 1.0)
    call_skill_update(category, modality, is_correct)

    data = {
        "data_id": data_id,
        "plan": plan,
        "question": question_text,
        "model_answer": model_answer,
        "ground_truth": ground_truth,
        "image_caption": image_caption,
        "used_memory_indices": used_memory_indices,
        "temp_messages": temp_messages,
        "judgement_nogt": judgement,
        "judgement_data": {
            **judgement_data,
            "skill_gated": skill_gated,
            "hard_gated": False,
            "skill_confidence": confidence,
            "skill_evidence_count": skill.get("evidence_count", 0) if skill else 0,
        },
    }
    get_user_response_from_url(memory_bank_save_url, data)

    log_inference_output({
        "data_id": data_id,
        "question": question_text,
        "ground_truth": ground_truth,
        "model_answer": model_answer,
        "plan": plan,
        "judgement_nogt": judgement,
        "judgement_data": judgement_data,
        "judgement_gt": judgement_gt,
        "reward": final_reward,
        "format_score": format_score,
        "accuracy_score": accuracy_score,
        "accuracy_score_gt": accuracy_score_gt,
        "reward_accuracy_score": reward_accuracy_score,
        "reward_source": actual_reward_source,
        "category": category,
        "modality": modality,
        "skill_gated": skill_gated,
        "hard_gated": False,
        "skill_confidence": confidence,
        "skill_evidence_count": skill.get("evidence_count", 0) if skill else 0,
        "executor_trace": executor_trace,
    })

    return final_reward


def get_user_response_from_url(url, data, max_retries=3, base_delay=1.0):
    last_exception = None
    for attempt in range(max_retries):
        try:
            response = requests.post(
                url, json=data, timeout=600,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            return response.json()
        except (requests.exceptions.Timeout,
                requests.exceptions.RequestException,
                ValueError) as e:
            last_exception = e
            attempt_num = attempt + 1
            if isinstance(e, requests.exceptions.Timeout):
                err_type = "TIMEOUT"
            elif isinstance(e, ValueError):
                err_type = "JSON_PARSE"
            else:
                err_type = "NETWORK"
            logger.warning(f"[{err_type}] Attempt {attempt_num}/{max_retries} failed for {url}: {e}")
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
            else:
                logger.error(f"All {max_retries} attempts failed for {url}")
    return "Default user response due to persistent network error"
