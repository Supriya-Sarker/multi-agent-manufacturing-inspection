# inspect_checkpoints.py
import json

MEMORY_PATH = "/Users/supriya/Documents/data/agentic_system/multi-agent-manufacturing-inspection/training/reviewer_logs/run_memory.json"

with open(MEMORY_PATH, "r") as f:
    memory = json.load(f)

print("=== Saved Memory from Last Run ===\n")

print("Focus Areas (injected into next vision agent run):")
if memory["focus_areas"]:
    for area in memory["focus_areas"]:
        print(f"  - {area}")
else:
    print("  none")

print("\nCategory Scores (from last frame):")
for category, score in memory["category_scores"].items():
    bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
    print(f"  {category:<25} {bar}  {score}")