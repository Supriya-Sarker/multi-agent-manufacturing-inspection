import json
from pathlib import Path

DSPy_DIR    = Path("/Users/supriya/Documents/data/agentic_system/multi-agent-manufacturing-inspection/training/dspy_optimized_prompts")
PROMPTS_DIR = Path("/Users/supriya/Documents/data/agentic_system/multi-agent-manufacturing-inspection/training/prompts")

strategies = ["zero_shot", "zero_shot_cot", "few_shot", "few_shot_cot"]

for strategy in strategies:
    json_path = DSPy_DIR / f"optimized_{strategy}.json"

    if not json_path.exists():
        print(f"Skipping {strategy} — JSON not found yet")
        continue

    with open(json_path) as f:
        data = json.load(f)

    # instruction = data["inspect"]["signature"]["instructions"]
    key = "inspect" if "inspect" in data else "inspect.predict"
    instruction = data[key]["signature"]["instructions"]
    out_path = PROMPTS_DIR / f"vision_prompt_dspy_{strategy}.txt"

    with open(out_path, "w") as f:
        f.write(instruction)

    print(f"✓ Saved → {out_path}")
    print(f"  Instruction preview: {instruction[:100]}...")
    print()

print("Done.")