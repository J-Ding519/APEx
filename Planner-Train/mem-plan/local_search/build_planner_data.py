#!/usr/bin/env python3
"""
将 skill_context 注入已有 Planner 训练 parquet。

用法:
    python build_planner_data.py \
        --input_parquet /path/to/fvqa_matpo_train_planner.parquet \
        --skill_repo /path/to/skill_repo.json \
        --output_parquet /path/to/planner_with_skill.parquet \
        --classify_url http://127.0.0.1:8002/v1
"""

import argparse
import json
import re
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

CATEGORIES = [
    "location", "human", "time", "career", "process",
    "definition", "event", "function", "property", "others",
]

CLASSIFY_PROMPT = """You are a text classifier. Given an input question, determine exactly one category.
Allowed Categories: location, human, time, career, process, definition, event, function, property.
Output only the category name, no explanation.

Question: {question}"""


def classify_question_llm(client, model, question: str) -> str:
    prompt = CLASSIFY_PROMPT.format(question=question[:200])
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
        return "others"
    except Exception as e:
        print(f"Classification error: {e}")
        return "others"


def load_skill_repo(path: str) -> dict:
    """加载 skill_repo.json，返回 {category_modality: skill_obj} 映射"""
    with open(path, "r", encoding="utf-8") as f:
        skills = json.load(f)
    skill_map = {}
    for skill in skills:
        key = f"{skill['category']}_{skill['modality']}"
        if key not in skill_map or skill.get("version", 0) > skill_map[key].get("version", 0):
            skill_map[key] = skill
    return skill_map


def format_skill_context(skill: dict) -> str:
    """将 skill JSON 格式化为 Planner 可读的文本"""
    lines = []
    if skill.get("description"):
        lines.append(f"**Strategy**: {skill['description']}")
    if skill.get("procedure"):
        lines.append("**Procedure**:")
        for step in skill["procedure"]:
            lines.append(f"  {step['step']}. {step['action']}")
            if step.get("purpose"):
                lines.append(f"     Purpose: {step['purpose']}")
    if skill.get("pitfalls"):
        lines.append("**Pitfalls to avoid**:")
        for p in skill["pitfalls"]:
            lines.append(f"  - {p}")
    if skill.get("win_rate") is not None:
        lines.append(f"**Historical win rate**: {skill['win_rate']:.0%}")
    return "\n".join(lines)


def extract_question(prompt_field) -> str:
    """从 parquet 的 prompt 字段提取问题文本"""
    if isinstance(prompt_field, list):
        for msg in prompt_field:
            if isinstance(msg, dict) and msg.get("role") == "user":
                return msg.get("content", "")
        return prompt_field[0].get("content", "") if prompt_field else ""
    return str(prompt_field)


def main():
    parser = argparse.ArgumentParser(description="Inject skill_context into Planner parquet")
    parser.add_argument("--input_parquet", required=True, help="原始 Planner 训练 parquet")
    parser.add_argument("--skill_repo", required=True, help="skill_repo.json 路径")
    parser.add_argument("--output_parquet", required=True, help="输出带 skill_context 的 parquet")
    parser.add_argument("--classify_url", required=True, help="LLM 分类服务地址 (OpenAI 兼容)")
    parser.add_argument("--classify_model", default="qwen", help="分类模型名")
    parser.add_argument("--max_workers", type=int, default=16, help="并发分类线程数")
    args = parser.parse_args()

    print(f"[1/4] 加载 skill_repo: {args.skill_repo}")
    skill_map = load_skill_repo(args.skill_repo)
    print(f"  已加载 {len(skill_map)} 个 skill: {list(skill_map.keys())}")

    print(f"[2/4] 读取 parquet: {args.input_parquet}")
    df = pd.read_parquet(args.input_parquet)
    print(f"  样本数: {len(df)}, 列: {list(df.columns)}")

    client = OpenAI(api_key="EMPTY", base_url=args.classify_url)

    # 如果已有 category 列，直接使用；否则调用 LLM 分类
    if "category" in df.columns and df["category"].notna().all():
        print("  已有 category 列，跳过分类")
        categories = df["category"].tolist()
    else:
        print(f"[3/4] 对 {len(df)} 条样本进行分类 (workers={args.max_workers})...")
        questions = [extract_question(row) for row in df["prompt"]]
        categories = [None] * len(questions)

        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_idx = {
                executor.submit(classify_question_llm, client, args.classify_model, q): i
                for i, q in enumerate(questions)
            }
            for future in tqdm(as_completed(future_to_idx), total=len(questions), desc="分类中"):
                idx = future_to_idx[future]
                categories[idx] = future.result()

        df["category"] = categories

    # 匹配 skill 并生成 skill_context
    # modality 归一化：parquet 中可能是 "image-text"，skill_repo 中是 "text-image"
    MODALITY_NORMALIZE = {
        "image-text": "text-image",
        "text-image": "text-image",
        "text-only": "text-only",
        "text_only": "text-only",
    }
    print("[4/4] 匹配 skill 并注入 skill_context...")
    skill_contexts = []
    match_count = 0
    for i, row in df.iterrows():
        category = categories[i] if isinstance(categories, list) else row.get("category", "others")
        modality_raw = row.get("modality", "text-only")
        modality = MODALITY_NORMALIZE.get(modality_raw, modality_raw)
        key = f"{category}_{modality}"
        skill = skill_map.get(key)
        if skill:
            skill_contexts.append(format_skill_context(skill))
            match_count += 1
        else:
            skill_contexts.append("No skill available for this question type.")

    df["skill_context"] = skill_contexts

    print(f"  匹配到 skill: {match_count}/{len(df)} ({match_count/len(df)*100:.1f}%)")

    # 统计各 category 分布
    cat_counts = df["category"].value_counts()
    print("  Category 分布:")
    for cat, cnt in cat_counts.items():
        key_text = f"{cat}_text-only"
        key_img = f"{cat}_text-image"
        has_skill = "✓" if key_text in skill_map or key_img in skill_map else "✗"
        print(f"    {cat}: {cnt} [{has_skill}]")

    df.to_parquet(args.output_parquet, index=False)
    print(f"\n  输出: {args.output_parquet}")
    print("  完成!")


if __name__ == "__main__":
    main()
