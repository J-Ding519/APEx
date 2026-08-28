"""Generate mock parquet data for Writer training.

Usage:
    python mock_data_gen.py --output_dir /path/to/output
"""
import argparse
import json
import random
import uuid

import pandas as pd

CATEGORIES = ["location", "human", "time", "career", "process", "definition", "event", "function", "property"]
MODALITIES = ["text-only", "text-image"]

SAMPLE_QUESTIONS = {
    "location": [
        "Where is the Palace of the Lost City located?",
        "What country is the Eiffel Tower in?",
        "Which city hosts the headquarters of the United Nations?",
    ],
    "human": [
        "Who directed the movie Inception?",
        "What is the full name of the author of Harry Potter?",
        "Who was the first person to walk on the moon?",
    ],
    "time": [
        "When was the Declaration of Independence signed?",
        "What year did World War II end?",
        "When was the first iPhone released?",
    ],
    "career": [
        "What is the occupation of Elon Musk?",
        "What did Marie Curie study?",
        "What field does Noam Chomsky work in?",
    ],
    "process": [
        "How is steel manufactured?",
        "What are the steps to make sourdough bread?",
        "How does photosynthesis work?",
    ],
    "definition": [
        "What is quantum entanglement?",
        "Define the term 'blockchain'.",
        "What does GDP stand for?",
    ],
    "event": [
        "What happened during the Boston Tea Party?",
        "Describe the events of the 2008 financial crisis.",
        "What caused the Chernobyl disaster?",
    ],
    "function": [
        "What is the function of the mitochondria?",
        "What does the liver do in the human body?",
        "What role does DNS play in the internet?",
    ],
    "property": [
        "What color is the Golden Gate Bridge?",
        "How tall is Mount Everest?",
        "What is the boiling point of water?",
    ],
}

SAMPLE_WORKFLOWS_CORRECT = [
    "1. Search entity name to identify context. 2. Search specific attribute with entity. 3. Cross-verify with second source. 4. Extract answer.",
    "1. Identify key entities in question. 2. Search for factual information. 3. Validate against multiple results. 4. Formulate concise answer.",
    "1. Break question into sub-queries. 2. Search each sub-query. 3. Combine findings. 4. Provide final answer.",
]

SAMPLE_WORKFLOWS_INCORRECT = [
    "1. Search with overly broad query. 2. Got irrelevant results. 3. Guessed answer without verification.",
    "1. Misidentified the entity. 2. Searched wrong topic. 3. Returned incorrect information.",
    "1. Used single vague search. 2. Did not cross-verify. 3. Answered based on partial information.",
]

SAMPLE_SKILL = {
    "skill_name": "location_text_search_strategy",
    "category": "location",
    "modality": "text-only",
    "description": "Strategy for answering location-based questions using text search.",
    "procedure": [
        {"step": 1, "action": "Search for the entity mentioned in the question", "purpose": "Identify the subject"},
        {"step": 2, "action": "Search entity + location attribute keyword", "purpose": "Find specific location info"},
        {"step": 3, "action": "Cross-verify with a second search query", "purpose": "Ensure accuracy"},
    ],
    "pitfalls": [
        "Using overly broad search queries that return irrelevant results",
        "Not cross-verifying location information from a single source",
    ],
    "win_rate": 0.75,
    "version": 1,
    "evidence_count": 20,
}


def generate_memory(category: str, correct: bool) -> dict:
    questions = SAMPLE_QUESTIONS.get(category, SAMPLE_QUESTIONS["location"])
    question = random.choice(questions)
    if len(question) > 50:
        question = question[:47] + "..."

    if correct:
        workflow = random.choice(SAMPLE_WORKFLOWS_CORRECT)
    else:
        workflow = random.choice(SAMPLE_WORKFLOWS_INCORRECT)
    if len(workflow) > 200:
        workflow = workflow[:197] + "..."

    return {
        "data_id": f"mem_{uuid.uuid4().hex[:8]}",
        "question": question,
        "workflow_summary": workflow,
        "judgement": "correct" if correct else "incorrect",
        "status": random.choice(["new", "anomaly"]) if not correct else "new",
    }


def generate_sample(category: str, modality: str, has_skill: bool) -> dict:
    num_memories = random.randint(5, 15)
    num_correct = random.randint(num_memories // 3, num_memories)
    memories = []
    for i in range(num_memories):
        correct = i < num_correct
        memories.append(generate_memory(category, correct))
    random.shuffle(memories)

    if has_skill:
        skill = dict(SAMPLE_SKILL)
        skill["category"] = category
        skill["modality"] = modality
        skill["skill_name"] = f"{category}_{modality.replace('-', '_')}_strategy"
        current_skill = json.dumps(skill)
    else:
        current_skill = "null"

    return {
        "data_id": f"writer_{uuid.uuid4().hex[:8]}",
        "category": category,
        "modality": modality,
        "current_skill": current_skill,
        "memories": json.dumps(memories),
        "reward_model": {"ground_truth": ""},
        "prompt": [{"role": "user", "content": "placeholder"}],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--train_samples_per_combo", type=int, default=8)
    parser.add_argument("--val_samples_per_combo", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    train_rows = []
    val_rows = []

    for category in CATEGORIES:
        for modality in MODALITIES:
            for i in range(args.train_samples_per_combo):
                has_skill = i >= args.train_samples_per_combo // 2
                train_rows.append(generate_sample(category, modality, has_skill))
            for i in range(args.val_samples_per_combo):
                has_skill = i >= args.val_samples_per_combo // 2
                val_rows.append(generate_sample(category, modality, has_skill))

    train_df = pd.DataFrame(train_rows)
    val_df = pd.DataFrame(val_rows)

    train_path = f"{args.output_dir}/writer_train_mock.parquet"
    val_path = f"{args.output_dir}/writer_val_mock.parquet"

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    print(f"Generated {len(train_rows)} train samples -> {train_path}")
    print(f"Generated {len(val_rows)} val samples -> {val_path}")
    print(f"Categories: {len(CATEGORIES)}, Modalities: {len(MODALITIES)}")
    print(f"Combinations: {len(CATEGORIES) * len(MODALITIES)}")


if __name__ == "__main__":
    main()
