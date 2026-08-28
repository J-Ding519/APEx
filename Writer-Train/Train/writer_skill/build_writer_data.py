"""Build Writer training data from Executor inference results.

Reads iter*.jsonl files (Executor inference output) and/or Memory-Serve
memory.json snapshots, classifies questions by category, groups memories
by (category, modality), and outputs Writer-format parquet files.

Usage:
    # From inference jsonl files (will call LLM classifier for category):
    python build_writer_data.py \
        --input_jsonl /path/to/iter1.jsonl \
        --output_dir /path/to/output \
        --classify_url http://127.0.0.1:8000/v1

    # From Memory-Serve saved json (already grouped by category/modality):
    python build_writer_data.py \
        --input_memory /path/to/memory.json \
        --output_dir /path/to/output

    # With existing skill repository for evolution samples:
    python build_writer_data.py \
        --input_jsonl /path/to/iter1.jsonl \
        --output_dir /path/to/output \
        --classify_url http://127.0.0.1:8000/v1 \
        --skill_repo /path/to/skill_repo.json
"""
import argparse
import json
import os
import random
import re
import uuid
from collections import defaultdict
from typing import Dict, List, Optional

import pandas as pd

CATEGORIES = [
    "location", "human", "time", "career", "process",
    "definition", "event", "function", "property", "others",
]
MODALITIES = ["text-only", "text-image"]

MAX_QUESTION_LEN = 50
MAX_WORKFLOW_LEN = 200
DEFAULT_BATCH_MIN = 5
DEFAULT_BATCH_MAX = 15
EVOLUTION_RATIO = 0.5


# ---------------------------------------------------------------------------
# LLM-based question classifier (mirrors Memory-Serve logic)
# ---------------------------------------------------------------------------

CLASSIFY_PROMPT = """You are a text classifier. Given an input question, determine exactly one category.
Allowed Categories: location, human, time, career, process, definition, event, function, property.
Output only the category name, no explanation.

Question: {question}"""


def classify_question_llm(question: str, client, model: str = "qwen") -> str:
    """Classify a question into a category using an LLM."""
    prompt = CLASSIFY_PROMPT.format(question=question)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=32,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        answer = resp.choices[0].message.content.strip().lower()
        answer = re.sub(r"[^a-z]", "", answer)
        if answer in CATEGORIES:
            return answer
    except Exception as e:
        print(f"[WARN] Classification failed: {e}")
    return "others"

def truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def determine_modality(image_caption) -> str:
    if isinstance(image_caption, list):
        return "text-image" if any(c.strip() for c in image_caption) else "text-only"
    if isinstance(image_caption, str):
        return "text-image" if image_caption.strip() else "text-only"
    return "text-only"


def extract_workflow_from_messages(question: str, messages: list) -> str:
    """Build a compact workflow summary from Executor messages."""
    trace = f"Question: {truncate(question, MAX_QUESTION_LEN)}\n"
    round_idx = 1
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                item.get("text", "") for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            )
        if role == "assistant":
            think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
            summary = think_match.group(1).strip()[:100] if think_match else content[:100]
            trace += f"{round_idx}. {summary}\n"
            round_idx += 1
    return truncate(trace, MAX_WORKFLOW_LEN)


# ---------------------------------------------------------------------------
# Source 1: Load from Executor inference jsonl
# ---------------------------------------------------------------------------

def load_from_jsonl(
    jsonl_path: str,
    client=None,
    model: str = "qwen",
) -> Dict[str, Dict[str, List[dict]]]:
    """Load inference results and group memories by (category, modality).

    Returns: {modality: {category: [memory_dict, ...]}}
    """
    store: Dict[str, Dict[str, List[dict]]] = {
        mod: {cat: [] for cat in CATEGORIES} for mod in MODALITIES
    }

    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    total = len(lines)
    for idx, line in enumerate(lines):
            item = json.loads(line)
            if "error" in item:
                continue

            question = item.get("question", "")
            judgement = item.get("judgement", "incorrect")
            if isinstance(judgement, bool):
                judgement = "correct" if judgement else "incorrect"
            image_caption = item.get("image_caption", "")
            modality = determine_modality(image_caption)
            messages = item.get("messages", [])

            if client is not None:
                category = classify_question_llm(question, client, model)
                if (idx + 1) % 50 == 0 or idx == total - 1:
                    print(f"  Classified {idx + 1}/{total} questions")
            else:
                category = "others"

            workflow = extract_workflow_from_messages(question, messages)

            mem = {
                "data_id": f"mem_{uuid.uuid4().hex[:8]}",
                "question": truncate(question, MAX_QUESTION_LEN),
                "workflow_summary": workflow,
                "judgement": judgement,
                "status": "new",
            }
            store[modality][category].append(mem)

    return store

# ---------------------------------------------------------------------------
# Source 2: Load from Memory-Serve saved json
# ---------------------------------------------------------------------------

def load_from_memory_json(
    memory_path: str,
) -> Dict[str, Dict[str, List[dict]]]:
    """Load memories from a Memory-Serve snapshot (already grouped)."""
    store: Dict[str, Dict[str, List[dict]]] = {
        mod: {cat: [] for cat in CATEGORIES} for mod in MODALITIES
    }

    with open(memory_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for modality in MODALITIES:
        if modality not in data:
            continue
        for category in CATEGORIES:
            if category not in data[modality]:
                continue
            for entry in data[modality][category]:
                mem = {
                    "data_id": entry.get("data_id", f"mem_{uuid.uuid4().hex[:8]}"),
                    "question": truncate(entry.get("question", ""), MAX_QUESTION_LEN),
                    "workflow_summary": truncate(
                        entry.get("workflow_summary", ""), MAX_WORKFLOW_LEN
                    ),
                    "judgement": entry.get("judgement", "incorrect"),
                    "status": "new",
                }
                store[modality][category].append(mem)

    return store


# ---------------------------------------------------------------------------
# Skill repository loader
# ---------------------------------------------------------------------------

def load_skill_repo(skill_repo_path: str) -> Dict[str, dict]:
    """Load skill repo keyed by '<category>_<modality>' -> skill dict."""
    if not skill_repo_path or not os.path.exists(skill_repo_path):
        return {}
    with open(skill_repo_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    repo = {}
    if isinstance(data, list):
        for skill in data:
            key = f"{skill.get('category', '')}_{skill.get('modality', '')}"
            repo[key] = skill
    elif isinstance(data, dict):
        for key, skill in data.items():
            repo[key] = skill
    return repo


# ---------------------------------------------------------------------------
# Batch memories into Writer training samples
# ---------------------------------------------------------------------------

def batch_memories(
    store: Dict[str, Dict[str, List[dict]]],
    skill_repo: Dict[str, dict],
    batch_min: int = DEFAULT_BATCH_MIN,
    batch_max: int = DEFAULT_BATCH_MAX,
    evolution_ratio: float = EVOLUTION_RATIO,
    seed: int = 42,
) -> List[dict]:
    """Split memories into batches and create Writer training samples."""
    rng = random.Random(seed)
    samples = []

    for modality in MODALITIES:
        for category in CATEGORIES:
            memories = store[modality][category]
            if len(memories) < batch_min:
                if memories:
                    samples.append(
                        _make_sample(category, modality, memories, skill_repo, rng, evolution_ratio)
                    )
                continue

            rng.shuffle(memories)
            idx = 0
            while idx < len(memories):
                remaining = len(memories) - idx
                if remaining < batch_min:
                    break
                batch_size = rng.randint(batch_min, min(batch_max, remaining))
                batch = memories[idx : idx + batch_size]
                samples.append(
                    _make_sample(category, modality, batch, skill_repo, rng, evolution_ratio)
                )
                idx += batch_size

    rng.shuffle(samples)
    return samples


def _make_sample(
    category: str,
    modality: str,
    memories: List[dict],
    skill_repo: Dict[str, dict],
    rng: random.Random,
    evolution_ratio: float,
) -> dict:
    key = f"{category}_{modality}"
    has_skill = key in skill_repo and rng.random() < evolution_ratio
    current_skill = json.dumps(skill_repo[key], ensure_ascii=False) if has_skill else "null"

    return {
        "data_id": f"writer_{uuid.uuid4().hex[:8]}",
        "category": category,
        "modality": modality,
        "current_skill": current_skill,
        "memories": json.dumps(memories, ensure_ascii=False),
        "reward_model": {"ground_truth": ""},
        "prompt": [{"role": "user", "content": "placeholder"}],
    }

# ---------------------------------------------------------------------------
# Merge multiple stores
# ---------------------------------------------------------------------------

def merge_stores(
    *stores: Dict[str, Dict[str, List[dict]]],
) -> Dict[str, Dict[str, List[dict]]]:
    merged: Dict[str, Dict[str, List[dict]]] = {
        mod: {cat: [] for cat in CATEGORIES} for mod in MODALITIES
    }
    for store in stores:
        for mod in MODALITIES:
            for cat in CATEGORIES:
                merged[mod][cat].extend(store.get(mod, {}).get(cat, []))
    return merged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build Writer training parquet from Executor results")
    parser.add_argument("--input_jsonl", type=str, nargs="*", default=[],
                        help="Executor inference jsonl files (iter*.jsonl)")
    parser.add_argument("--input_memory", type=str, default=None,
                        help="Memory-Serve saved json file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for parquet files")
    parser.add_argument("--classify_url", type=str, default=None,
                        help="vLLM/sglang endpoint for question classification")
    parser.add_argument("--classify_model", type=str, default="qwen",
                        help="Model name for classification")
    parser.add_argument("--skill_repo", type=str, default=None,
                        help="Path to skill repository json for evolution samples")
    parser.add_argument("--batch_min", type=int, default=DEFAULT_BATCH_MIN)
    parser.add_argument("--batch_max", type=int, default=DEFAULT_BATCH_MAX)
    parser.add_argument("--evolution_ratio", type=float, default=EVOLUTION_RATIO)
    parser.add_argument("--val_ratio", type=float, default=0.2,
                        help="Fraction of samples for validation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Build LLM client if needed
    client = None
    if args.classify_url and args.input_jsonl:
        from openai import OpenAI
        import httpx
        os.environ.setdefault("no_proxy", "127.0.0.1,localhost")
        client = OpenAI(
            api_key="EMPTY",
            base_url=args.classify_url,
            http_client=httpx.Client(
                timeout=httpx.Timeout(60.0, connect=10.0),
            ),
        )

    # Load data from all sources
    all_stores = []

    for jsonl_path in args.input_jsonl:
        print(f"Loading from jsonl: {jsonl_path}")
        store = load_from_jsonl(jsonl_path, client=client, model=args.classify_model)
        all_stores.append(store)

    if args.input_memory:
        print(f"Loading from memory json: {args.input_memory}")
        store = load_from_memory_json(args.input_memory)
        all_stores.append(store)

    if not all_stores:
        print("Error: No input data provided. Use --input_jsonl or --input_memory.")
        return

    merged = merge_stores(*all_stores) if len(all_stores) > 1 else all_stores[0]

    # Print statistics
    total = 0
    for mod in MODALITIES:
        for cat in CATEGORIES:
            count = len(merged[mod][cat])
            if count > 0:
                print(f"  {mod}/{cat}: {count} memories")
            total += count
    print(f"Total memories loaded: {total}")

    # Load skill repo
    skill_repo = load_skill_repo(args.skill_repo)
    if skill_repo:
        print(f"Loaded {len(skill_repo)} skills from repo")

    # Batch into samples
    samples = batch_memories(
        merged, skill_repo,
        batch_min=args.batch_min,
        batch_max=args.batch_max,
        evolution_ratio=args.evolution_ratio,
        seed=args.seed,
    )
    print(f"Generated {len(samples)} Writer training samples")

    if not samples:
        print("No samples generated. Check that input data has enough memories.")
        return

    # Train/val split
    rng = random.Random(args.seed)
    rng.shuffle(samples)
    val_count = max(1, int(len(samples) * args.val_ratio))
    val_samples = samples[:val_count]
    train_samples = samples[val_count:]

    # Save
    train_path = os.path.join(args.output_dir, "writer_train.parquet")
    val_path = os.path.join(args.output_dir, "writer_val.parquet")

    pd.DataFrame(train_samples).to_parquet(train_path, index=False)
    pd.DataFrame(val_samples).to_parquet(val_path, index=False)

    synthesis_count = sum(1 for s in samples if s["current_skill"] == "null")
    evolution_count = len(samples) - synthesis_count

    print(f"Train: {len(train_samples)} samples -> {train_path}")
    print(f"Val:   {len(val_samples)} samples -> {val_path}")
    print(f"Synthesis samples: {synthesis_count}, Evolution samples: {evolution_count}")


if __name__ == "__main__":
    main()
