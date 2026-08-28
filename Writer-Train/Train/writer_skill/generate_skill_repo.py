"""Generate skill_repo.json using trained Writer model.

Loads memories grouped by (category, modality), constructs Writer prompts
with multi-round evolution: first batch uses synthesis mode, subsequent
batches use evolution mode to iteratively refine the skill.

Usage:
    # From memory.json (already grouped, no classifier needed):
    python generate_skill_repo.py \
        --input_memory /path/to/memory.json \
        --model_url http://localhost:8000/v1 \
        --model_name writer \
        --output /path/to/skill_repo.json

    # From iter1.jsonl (needs classifier service):
    python generate_skill_repo.py \
        --input_jsonl /path/to/iter1.jsonl \
        --model_url http://localhost:8000/v1 \
        --model_name writer \
        --output /path/to/skill_repo.json \
        --classify_url http://localhost:8002/v1 \
        --classify_model qwen
"""
import argparse
import json
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from prompt import build_writer_prompt
from build_writer_data import (
    CATEGORIES,
    MODALITIES,
    load_from_memory_json,
    load_from_jsonl,
)
from typing import Optional


def parse_skill_json(solution_str: str) -> tuple:
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


def generate_skill_repo(args):
    # Load memories
    if args.input_memory:
        print(f"Loading memories from: {args.input_memory}")
        store = load_from_memory_json(args.input_memory)
    elif args.input_jsonl:
        print(f"Loading memories from: {args.input_jsonl}")
        classify_client = None
        if args.classify_url:
            classify_client = OpenAI(api_key="EMPTY", base_url=args.classify_url)
        for jsonl_path in args.input_jsonl:
            store = load_from_jsonl(
                jsonl_path,
                client=classify_client,
                model=args.classify_model,
            )
    else:
        print("[Error] Must provide --input_memory or --input_jsonl")
        sys.exit(1)

    # Print stats
    print("\nMemory statistics:")
    total = 0
    for modality in MODALITIES:
        for category in CATEGORIES:
            count = len(store[modality][category])
            if count > 0:
                print(f"  {modality}/{category}: {count}")
                total += count
    print(f"  Total: {total}\n")

    # Initialize Writer model client
    client = OpenAI(api_key="EMPTY", base_url=args.model_url)

    skill_repo = []
    failed = []

    for modality in MODALITIES:
        for category in CATEGORIES:
            memories = store[modality][category]
            if len(memories) < args.min_memories:
                if len(memories) > 0:
                    print(f"  [Skip] {modality}/{category}: only {len(memories)} memories (min={args.min_memories})")
                continue

            # Split memories into batches
            batches = [memories[i:i + args.batch_size] for i in range(0, len(memories), args.batch_size)]
            max_rounds = min(len(batches), args.max_rounds)

            print(f"  [{modality}/{category}] {len(memories)} memories, {max_rounds} rounds")

            current_skill = None

            for round_idx in range(max_rounds):
                batch = batches[round_idx]
                is_first = (round_idx == 0)

                if is_first:
                    skill_input = "null"
                    mode = "synthesize"
                else:
                    skill_input = json.dumps(current_skill, ensure_ascii=False)
                    mode = "evolution"

                messages = build_writer_prompt(category, modality, skill_input, batch)

                print(f"    Round {round_idx + 1}/{max_rounds} ({mode}, {len(batch)} memories)...", end=" ")

                try:
                    response = client.chat.completions.create(
                        model=args.model_name,
                        messages=messages,
                        max_tokens=args.max_tokens,
                        temperature=0.6,
                        top_p=0.95,
                    )
                    content = response.choices[0].message.content
                    parsed, valid = parse_skill_json(content)

                    if valid and parsed and "skill" in parsed:
                        operation = parsed.get("operation", "synthesize")
                        skill = parsed["skill"]
                        skill["category"] = category
                        skill["modality"] = modality

                        if operation == "skip" and current_skill:
                            print(f"skip (no update needed)")
                        elif operation == "create" and current_skill:
                            # Writer says new strategy needed — keep both
                            skill_repo.append(skill)
                            print(f"create (new complementary skill)")
                        else:
                            current_skill = skill
                            print(f"OK (win_rate={skill.get('win_rate', '?')})")
                    else:
                        print("FAILED (invalid JSON)")
                        if args.verbose and content:
                            print(f"      Raw: {content[:200]}...")
                        if is_first:
                            break  # Can't continue without initial skill
                except Exception as e:
                    print(f"ERROR: {e}")
                    if is_first:
                        break

            if current_skill:
                skill_repo.append(current_skill)
            else:
                failed.append(f"{modality}/{category}")

    # Save results
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(skill_repo, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"  Generated {len(skill_repo)} skills")
    if failed:
        print(f"  Failed: {len(failed)} ({', '.join(failed)})")
    print(f"  Output: {args.output}")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description="Generate skill_repo.json using trained Writer model")
    parser.add_argument("--input_memory", type=str, help="Path to memory.json (Memory-Serve snapshot)")
    parser.add_argument("--input_jsonl", type=str, nargs="+", help="Path to iter*.jsonl files")
    parser.add_argument("--model_url", type=str, required=True, help="Writer model vLLM API URL")
    parser.add_argument("--model_name", type=str, default="writer", help="Served model name")
    parser.add_argument("--output", type=str, required=True, help="Output skill_repo.json path")
    parser.add_argument("--batch_size", type=int, default=15, help="Max memories per prompt")
    parser.add_argument("--max_rounds", type=int, default=5, help="Max evolution rounds per group")
    parser.add_argument("--min_memories", type=int, default=3, help="Min memories to generate skill")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max generation tokens")
    parser.add_argument("--classify_url", type=str, help="LLM classifier URL (for jsonl input)")
    parser.add_argument("--classify_model", type=str, default="qwen", help="Classifier model name")
    parser.add_argument("--verbose", action="store_true", help="Print raw outputs on failure")
    args = parser.parse_args()

    generate_skill_repo(args)


if __name__ == "__main__":
    main()
