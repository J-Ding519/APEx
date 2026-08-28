"""
Skill Repository for Skill-Guided Adaptive TTRL.

Uses the trained Writer model (Qwen3-8B, GRPO step 40) for skill synthesis
and evolution. Skills follow the Writer's structured JSON format:
  - procedure: list of {step, action, purpose}
  - pitfalls: list of strings
  - win_rate, evidence_count, version, etc.

Loads from Writer-generated skill_repo.json (list format) or from
per-(category, modality) dict format. During TTRL training, stats are
updated online and the Writer is called for periodic skill evolution.
"""
import json
import os
import re
import threading
import logging
from typing import Dict, List, Optional, Any
from openai import OpenAI

logger = logging.getLogger(__name__)

CATEGORIES = ["location", "human", "time", "career", "process",
              "definition", "event", "function", "property", "others"]
MODALITIES = ["text-only", "text-image"]

EVIDENCE_THRESHOLD = 10

# ============================================================
# Writer prompts (same as Writer-Train/Train/writer_skill/prompt.py)
# ============================================================

SKILL_OUTPUT_SCHEMA = """{
  "operation": "synthesize | refine | create | skip",
  "skill": {
    "skill_name": "<category>_<modality>_strategy",
    "category": "<category>",
    "modality": "<modality>",
    "description": "One-sentence description of the skill",
    "procedure": [
      {"step": 1, "action": "Concrete action description", "purpose": "Why this step matters"}
    ],
    "pitfalls": ["Common mistake to avoid"],
    "win_rate": 0.0,
    "version": 1,
    "evidence_count": 0
  },
  "absorbed_ids": ["list of memory data_ids whose patterns are now captured in the skill"]
}"""

WRITER_SYSTEM_PROMPT = (
    "You are a skill synthesis and evolution expert in a memory-augmented agent system. "
    "Your job is to analyze batches of task execution memories and distill them into "
    "reusable, structured skill documents. Skills encode proven strategies and common "
    "pitfalls for specific task categories. Output valid JSON only."
)

SKILL_SYNTHESIS_PROMPT = """Analyze the following batch of task memories from the "{category}" category ({modality} modality) and synthesize a reusable skill document.

[Memories]
{formatted_memories}

Requirements:
1. Identify common strategies in successful (correct) memories — what search patterns, reasoning steps, or tool usage led to correct answers.
2. Identify common failure modes in incorrect memories — what mistakes, wrong assumptions, or missing steps caused failures.
3. Synthesize a structured skill with clear, atomic procedure steps that generalize across these memories.
4. List specific pitfalls to avoid, grounded in observed failure cases.
5. Compute win_rate as the ratio of correct memories to total memories.
6. Set operation to "synthesize", version to 1, evidence_count to the number of memories.

Output a JSON object with this exact structure (no extra text before or after the JSON):
{skill_output_schema}"""

SKILL_EVOLUTION_PROMPT = """Given the current skill and a batch of new evidence memories, decide how to update the skill.

[Current Skill]
{current_skill}

[New Evidence Memories]
{formatted_memories}

Choose ONE operation:
- "refine": Failures expose a specific defect in the current procedure. Fix the defective steps and update pitfalls, but preserve steps validated by successes (conservative editing — do not break what works).
- "create": Memories reveal an effective strategy pattern NOT covered by the current skill. Create a new skill.
- "skip": New evidence is insufficient to justify changes. Keep the skill unchanged, only update win_rate and evidence_count.

Conservative editing principles:
- Successful memories define invariants — the procedure steps they validate must NOT be modified.
- Failed memories define modification targets — only change steps related to the failure cause.
- Never remove a pitfall that is still relevant.

Output a JSON object with this exact structure (no extra text before or after the JSON):
{skill_output_schema}"""


def format_memories_for_writer(memories: List[Dict]) -> str:
    lines = []
    for i, mem in enumerate(memories, 1):
        judgement = mem.get("judgement_nogt", mem.get("judgement", "unknown"))
        lines.append(
            f"Memory {i} [{judgement.upper()}]:\n"
            f"  Question: {mem.get('question', '')}\n"
            f"  Workflow: {mem.get('workflow_summary', 'N/A')}"
        )
    return "\n\n".join(lines)


def parse_skill_json(text: str) -> Optional[dict]:
    cleaned = text.strip()
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1].strip()
    json_match = re.search(r'\{[\s\S]*\}', cleaned)
    if not json_match:
        return None
    try:
        parsed = json.loads(json_match.group())
        if "skill" in parsed:
            return parsed
    except json.JSONDecodeError:
        pass
    return None


# ============================================================
# SkillEntry — mirrors Writer output format
# ============================================================

class SkillEntry:
    def __init__(self, category: str, modality: str,
                 skill_name: str = "",
                 description: str = "",
                 procedure: Optional[List[Dict[str, Any]]] = None,
                 pitfalls: Optional[List[str]] = None,
                 win_rate: float = 0.0,
                 evidence_count: int = 0,
                 correct_count: int = 0,
                 version: int = 0):
        self.category = category
        self.modality = modality
        self.skill_name = skill_name or f"{category}_{modality}_strategy"
        self.description = description
        self.procedure = procedure or []
        self.pitfalls = pitfalls or []
        self.win_rate = win_rate
        self.evidence_count = evidence_count
        self.correct_count = correct_count
        self.version = version

    @property
    def procedure_text(self) -> str:
        if not self.procedure:
            return ""
        lines = []
        for step in self.procedure:
            if isinstance(step, dict):
                n = step.get("step", "")
                action = step.get("action", "")
                purpose = step.get("purpose", "")
                lines.append(f"{n}. {action} — {purpose}" if purpose else f"{n}. {action}")
            else:
                lines.append(str(step))
        return "\n".join(lines)

    @property
    def pitfalls_text(self) -> str:
        if not self.pitfalls:
            return ""
        return "\n".join(f"- {p}" for p in self.pitfalls)

    def to_dict(self) -> dict:
        return {
            "skill_name": self.skill_name,
            "category": self.category,
            "modality": self.modality,
            "description": self.description,
            "procedure": self.procedure,
            "pitfalls": self.pitfalls,
            "win_rate": round(self.win_rate, 4),
            "evidence_count": self.evidence_count,
            "correct_count": self.correct_count,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SkillEntry":
        return cls(
            category=d.get("category", "others"),
            modality=d.get("modality", "text-only"),
            skill_name=d.get("skill_name", ""),
            description=d.get("description", ""),
            procedure=d.get("procedure", []),
            pitfalls=d.get("pitfalls", []),
            win_rate=d.get("win_rate", 0.0),
            evidence_count=d.get("evidence_count", 0),
            correct_count=d.get("correct_count", 0),
            version=d.get("version", 0),
        )


# ============================================================
# SkillRepository
# ============================================================

class SkillRepository:
    def __init__(self, evidence_threshold: int = EVIDENCE_THRESHOLD):
        self.evidence_threshold = evidence_threshold
        self._lock = threading.Lock()
        self.skills: Dict[str, Dict[str, SkillEntry]] = {
            modality: {
                cat: SkillEntry(category=cat, modality=modality)
                for cat in CATEGORIES
            }
            for modality in MODALITIES
        }

    def match_skill(self, category: str, modality: str) -> Optional[SkillEntry]:
        if category not in CATEGORIES:
            category = "others"
        if modality not in MODALITIES:
            modality = "text-only"
        return self.skills[modality][category]

    def compute_confidence(self, skill: Optional[SkillEntry]) -> float:
        if skill is None or skill.evidence_count == 0:
            return 0.0
        reliability = min(skill.evidence_count / self.evidence_threshold, 1.0)
        return skill.win_rate * reliability

    def match_and_confidence(self, category: str, modality: str) -> tuple:
        skill = self.match_skill(category, modality)
        confidence = self.compute_confidence(skill)
        return skill, confidence

    def update_stats(self, category: str, modality: str, is_correct: bool):
        if category not in CATEGORIES:
            category = "others"
        if modality not in MODALITIES:
            modality = "text-only"
        with self._lock:
            skill = self.skills[modality][category]
            skill.evidence_count += 1
            if is_correct:
                skill.correct_count += 1
            skill.win_rate = skill.correct_count / skill.evidence_count

    def evolve_skills_with_writer(self, memory_processor, writer_client: OpenAI,
                                  writer_model: str = "writer",
                                  batch_size: int = 15, max_tokens: int = 2048):
        """Call the trained Writer model to synthesize/evolve skills from memory.

        This is the correct way to generate skills — uses the GRPO-trained Writer
        rather than ad-hoc LLM prompts.
        """
        for modality in MODALITIES:
            for cat in CATEGORIES:
                bucket = memory_processor.memory_store[modality][cat]
                if bucket.get_size() < 3:
                    continue

                memories = []
                for entry in bucket.memory_data:
                    judgement = entry.get("judgement_nogt", "unknown")
                    memories.append({
                        "question": entry.get("question", ""),
                        "workflow_summary": entry.get("workflow_summary", ""),
                        "judgement_nogt": judgement,
                        "data_id": entry.get("data_id", ""),
                    })

                # MemoryBucket appends new entries, so the tail contains the
                # most recent evidence gathered since earlier evolutions.
                batch = memories[-batch_size:]
                formatted = format_memories_for_writer(batch)
                current_skill = self.skills[modality][cat]

                if current_skill.version == 0 or not current_skill.procedure:
                    user_content = SKILL_SYNTHESIS_PROMPT.format(
                        category=cat,
                        modality=modality,
                        formatted_memories=formatted,
                        skill_output_schema=SKILL_OUTPUT_SCHEMA,
                    )
                    mode = "synthesize"
                else:
                    user_content = SKILL_EVOLUTION_PROMPT.format(
                        current_skill=json.dumps(current_skill.to_dict(), ensure_ascii=False),
                        formatted_memories=formatted,
                        skill_output_schema=SKILL_OUTPUT_SCHEMA,
                    )
                    mode = "evolution"

                messages = [
                    {"role": "system", "content": WRITER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ]

                try:
                    response = writer_client.chat.completions.create(
                        model=writer_model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=0.6,
                        top_p=0.95,
                    )
                    content = response.choices[0].message.content
                    parsed = parse_skill_json(content)

                    if parsed and "skill" in parsed:
                        operation = parsed.get("operation", "synthesize")
                        new_skill_data = parsed["skill"]
                        new_skill_data["category"] = cat
                        new_skill_data["modality"] = modality

                        if operation == "skip":
                            skill = self.skills[modality][cat]
                            skill.evidence_count = new_skill_data.get("evidence_count", skill.evidence_count)
                            skill.win_rate = new_skill_data.get("win_rate", skill.win_rate)
                            logger.info(f"[Writer/{mode}] {modality}/{cat}: skip (no update needed)")
                        else:
                            new_entry = SkillEntry.from_dict(new_skill_data)
                            new_entry.correct_count = self.skills[modality][cat].correct_count
                            with self._lock:
                                self.skills[modality][cat] = new_entry
                            logger.info(
                                f"[Writer/{mode}] {modality}/{cat}: {operation} "
                                f"(version={new_entry.version}, "
                                f"procedures={len(new_entry.procedure)}, "
                                f"pitfalls={len(new_entry.pitfalls)})"
                            )
                    else:
                        logger.warning(f"[Writer/{mode}] {modality}/{cat}: invalid JSON output")
                except Exception as e:
                    logger.error(f"[Writer/{mode}] {modality}/{cat}: error - {e}")

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {}
        for modality in MODALITIES:
            data[modality] = {}
            for cat in CATEGORIES:
                data[modality][cat] = self.skills[modality][cat].to_dict()
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved skill repository to {path}")

    def load(self, path: str):
        """Load skills — supports both formats:
        1. Writer-generated list format (skill_repo.json): [{skill_name, category, modality, ...}]
        2. Per-(modality, category) dict format: {modality: {category: {...}}}
        """
        if not os.path.exists(path):
            logger.warning(f"Skill file not found: {path}")
            return
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            # Writer-generated format: list of skill entries
            loaded = 0
            for skill_data in data:
                cat = skill_data.get("category", "others")
                mod = skill_data.get("modality", "text-only")
                if cat not in CATEGORIES:
                    cat = "others"
                if mod not in MODALITIES:
                    mod = "text-only"
                self.skills[mod][cat] = SkillEntry.from_dict(skill_data)
                loaded += 1
            logger.info(f"Loaded {loaded} skills from Writer format: {path}")
        elif isinstance(data, dict):
            # Dict format: {modality: {category: {...}}}
            loaded = 0
            for modality in MODALITIES:
                if modality not in data:
                    continue
                for cat in CATEGORIES:
                    if cat not in data[modality]:
                        continue
                    self.skills[modality][cat] = SkillEntry.from_dict(data[modality][cat])
                    loaded += 1
            logger.info(f"Loaded {loaded} skills from dict format: {path}")

    def get_all_stats(self) -> List[Dict[str, Any]]:
        stats = []
        for modality in MODALITIES:
            for cat in CATEGORIES:
                skill = self.skills[modality][cat]
                if skill.evidence_count > 0 or skill.procedure:
                    stats.append({
                        "modality": modality,
                        "category": cat,
                        "skill_name": skill.skill_name,
                        "win_rate": round(skill.win_rate, 4),
                        "evidence_count": skill.evidence_count,
                        "confidence": round(self.compute_confidence(skill), 4),
                        "version": skill.version,
                        "procedures": len(skill.procedure),
                        "pitfalls": len(skill.pitfalls),
                    })
        return stats
