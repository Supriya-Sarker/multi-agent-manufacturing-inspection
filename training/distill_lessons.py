"""
distill_lessons.py

Reads all frame state JSONs from the memory training run and distills
actionable lessons for the vision agent's test phase prompt.

Run once after completing the full memory training run.
"""

import json
from pathlib import Path
from collections import defaultdict

# Paths
BASE_DIR          = Path("/Users/supriya/Documents/data/agentic_system/multi-agent-manufacturing-inspection/training")
MEMORY_STATES_DIR = BASE_DIR / "results" / "train" / "memory_states"
OUTPUT_PATH       = BASE_DIR / "prompts" / "lesson_note.txt"

# Load all state JSONs 
state_files = sorted(MEMORY_STATES_DIR.glob("*_state.json"))
print(f"Total state files: {len(state_files)}")

# Collect focus areas injected during training per category
focus_texts = defaultdict(list)

# Generic phrases to exclude — not actionable
GENERIC = {
    "Re-check using only the applicable Domain Decision Rules for this category.",
    "Re-check using the applicable Domain Decision Rules for this category.",
}

for path in state_files:
    with open(path) as f:
        data = json.load(f)

    for area in data.get("focus_areas", []):
        if ":" in area:
            cat, text = area.split(":", 1)
            text = text.strip()
            if text not in GENERIC:
                focus_texts[cat.strip()].append(text)

# Build lesson note — actionable only 
lesson_lines = []
lesson_lines.append("LESSONS LEARNED FROM TRAINING")
lesson_lines.append("="*50)
lesson_lines.append(
    "The following corrections were identified during training. "
    "Apply them carefully when predicting on new frames."
)
lesson_lines.append("")

for cat, texts in sorted(focus_texts.items()):
    unique_texts = list(dict.fromkeys(texts))  # deduplicate preserving order
    if not unique_texts:
        continue
    lesson_lines.append(f"[{cat.upper()}]")
    for t in unique_texts:
        lesson_lines.append(f"- {t}")
    lesson_lines.append("")

lesson_note = "\n".join(lesson_lines)

# Save 
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    f.write(lesson_note)

print("\nLesson note:")
print(lesson_note)
print(f"\nSaved → {OUTPUT_PATH}")