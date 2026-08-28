import json
import logging
import os
import re

from openai import OpenAI
import verl.utils.torch_functional as verl_F
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.model import compute_position_id_with_mask

from writer_skill.prompt import build_writer_prompt, format_memories, SKILL_JUDGE_PROMPT

logger = logging.getLogger(__name__)

openai_api_key = "EMPTY"
openai_api_base = os.getenv("JUDGE_URL")
client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
model_name = "qwen"

VALID_OPERATIONS = {"synthesize", "refine", "create", "skip"}
REQUIRED_SKILL_FIELDS = {"skill_name", "category", "modality", "description", "procedure", "pitfalls", "win_rate", "version", "evidence_count"}


class WriterRLHFDataset(RLHFDataset):
    def __getitem__(self, item):
        row_dict: dict = self.dataframe[item]

        category = row_dict["category"]
        modality = row_dict["modality"]
        current_skill = row_dict.get("current_skill", "null")
        memories_raw = row_dict.get("memories", "[]")

        if isinstance(memories_raw, str):
            memories = json.loads(memories_raw)
        else:
            memories = memories_raw

        messages = build_writer_prompt(category, modality, current_skill, memories)
        row_dict[self.prompt_key] = messages

        messages_for_tokenize = self._build_messages(row_dict)

        raw_prompt = self.tokenizer.apply_chat_template(
            messages_for_tokenize, add_generation_prompt=True, tokenize=False
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
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length:]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[:self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} exceeds {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids

        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages_for_tokenize

        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = {}

        row_dict["extra_info"]["category"] = category
        row_dict["extra_info"]["modality"] = modality
        row_dict["extra_info"]["current_skill"] = current_skill
        row_dict["extra_info"]["memories"] = memories_raw
        row_dict["extra_info"]["data_id"] = row_dict.get("data_id", "")

        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt

        row_dict["agent_name"] = "single_turn_agent"
        if "data_source" not in row_dict:
            row_dict["data_source"] = "writer_skill"
        return row_dict


def _parse_skill_json(solution_str: str) -> tuple[dict | None, bool]:
    cleaned = solution_str.strip()
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1].strip()

    json_match = re.search(r'\{[\s\S]*\}', cleaned)
    if not json_match:
        return None, False
    try:
        parsed = json.loads(json_match.group())
        return parsed, True
    except json.JSONDecodeError:
        return None, False


def _score_format(parsed: dict | None, is_valid_json: bool) -> float:
    if not is_valid_json or parsed is None:
        return 0.0
    score = 0.5
    if parsed.get("operation") in VALID_OPERATIONS:
        score += 0.25
    if "skill" in parsed:
        score += 0.25
    return score


def _score_skill_quality(parsed: dict | None) -> float:
    if parsed is None or "skill" not in parsed:
        return 0.0
    skill = parsed["skill"]
    if not isinstance(skill, dict):
        return 0.0

    score = 0.0
    present_fields = set(skill.keys()) & REQUIRED_SKILL_FIELDS
    score += 0.3 * (len(present_fields) / len(REQUIRED_SKILL_FIELDS))

    procedure = skill.get("procedure", [])
    if isinstance(procedure, list) and len(procedure) > 0:
        valid_steps = sum(
            1 for s in procedure
            if isinstance(s, dict) and "step" in s and "action" in s and "purpose" in s
        )
        score += 0.3 * (valid_steps / len(procedure))
    pitfalls = skill.get("pitfalls", [])
    if isinstance(pitfalls, list) and len(pitfalls) > 0:
        score += 0.2

    win_rate = skill.get("win_rate")
    if isinstance(win_rate, (int, float)) and 0.0 <= win_rate <= 1.0:
        score += 0.1
    if isinstance(skill.get("evidence_count"), int) and skill["evidence_count"] > 0:
        score += 0.1

    return score


def _score_llm_quality(solution_str: str, extra_info: dict | None) -> float:
    parsed, is_valid_json = _parse_skill_json(solution_str)
    if not is_valid_json or parsed is None:
        return 0.0

    memories_raw = extra_info.get("memories", "[]") if extra_info else "[]"
    if isinstance(memories_raw, str):
        try:
            memories = json.loads(memories_raw)
        except json.JSONDecodeError:
            memories = []
    else:
        memories = memories_raw

    current_skill = extra_info.get("current_skill", "null") if extra_info else "null"
    formatted_memories = format_memories(memories)
    generated_skill = json.dumps(parsed, ensure_ascii=False, indent=2)

    user_prompt = SKILL_JUDGE_PROMPT.format(
        memories=formatted_memories,
        current_skill=current_skill,
        generated_skill=generated_skill,
    )

    try:
        chat_response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": user_prompt}],
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            temperature=0.1,
        )
        response = chat_response.choices[0].message.content.strip()
        response = response.replace("\\n", "\n").replace("\\r", "\r")
        cleaned = re.sub(r"<think>.*?</think>|<think>|</think>", "", response, flags=re.DOTALL | re.IGNORECASE).strip()

        if cleaned == "A":
            return 1.0
        elif cleaned == "B":
            return 0.5
        elif cleaned == "C":
            return 0.0
        else:
            return 0.0
    except Exception as e:
        logger.warning(f"LLM Judge call failed: {e}, falling back to rule-based scoring")
        return _score_skill_quality(parsed)


def _score_evolution_rationality(parsed: dict | None, extra_info: dict | None) -> float:
    if parsed is None:
        return 0.0

    operation = parsed.get("operation", "")
    current_skill_str = extra_info.get("current_skill", "null") if extra_info else "null"
    is_synthesis = current_skill_str is None or current_skill_str == "null" or current_skill_str.strip() == ""

    if is_synthesis:
        return 1.0 if operation == "synthesize" else 0.3

    if operation == "refine":
        skill = parsed.get("skill", {})
        if not isinstance(skill, dict):
            return 0.2
        try:
            old_skill = json.loads(current_skill_str)
        except (json.JSONDecodeError, TypeError):
            return 0.5
        old_procedures = {s.get("action", "") for s in old_skill.get("procedure", []) if isinstance(s, dict)}
        new_procedures = {s.get("action", "") for s in skill.get("procedure", []) if isinstance(s, dict)}
        if old_procedures and new_procedures:
            preserved = len(old_procedures & new_procedures) / len(old_procedures)
            changed = 1.0 - preserved
            if 0.1 <= changed <= 0.5:
                return 0.9
            elif changed > 0.5:
                return 0.4
            else:
                return 0.7
        return 0.5

    elif operation == "create":
        return 0.7

    elif operation == "skip":
        skill = parsed.get("skill", {})
        if isinstance(skill, dict):
            try:
                old_skill = json.loads(current_skill_str)
            except (json.JSONDecodeError, TypeError):
                return 0.5
            old_proc = old_skill.get("procedure", [])
            new_proc = skill.get("procedure", [])
            if json.dumps(old_proc, sort_keys=True) == json.dumps(new_proc, sort_keys=True):
                return 0.9
            return 0.4
        return 0.5

    return 0.2


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info=None) -> float:
    parsed, is_valid_json = _parse_skill_json(solution_str)

    format_s = _score_format(parsed, is_valid_json)
    llm_quality_s = _score_llm_quality(solution_str, extra_info)
    evolution_s = _score_evolution_rationality(parsed, extra_info)

    final_score = 0.1 * format_s + 0.5 * llm_quality_s + 0.4 * evolution_s
    return final_score
