import dspy
import os
import re
import json
import pandas as pd
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate import Evaluate
from sklearn.model_selection import KFold
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
os.environ["GEMINI_API_KEY"] = os.getenv("GOOGLE_API_KEY")
lm = dspy.LM("gemini/gemini-flash-latest")
dspy.configure(lm=lm)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR      = Path("/Users/supriya/Documents/data/agentic_system/multi-agent-manufacturing-inspection/training")
IMAGES_DIR    = BASE_DIR / "images"
TRAIN_DIR     = IMAGES_DIR / "train_dataset"
FEW_SHOT_DIR  = IMAGES_DIR / "few_shot"

TRAIN_OVERLAY  = TRAIN_DIR / "overlay_images"
TRAIN_METADATA = TRAIN_DIR / "metadata"
TRAIN_SIGNAL   = TRAIN_DIR / "signal_plots"

FEW_SHOT_OVERLAY  = FEW_SHOT_DIR / "overlay_images"
FEW_SHOT_METADATA = FEW_SHOT_DIR / "metadata"
FEW_SHOT_SIGNAL   = FEW_SHOT_DIR / "signal_plots"

GT_CSV            = IMAGES_DIR / "images_ground_truth.csv"
DOMAIN_RULES_PATH = BASE_DIR / "prompts" / "domain_rules_1.txt"
OUTPUT_DIR        = BASE_DIR / "dspy_optimized_prompts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load domain rules ──────────────────────────────────────────────────────────
with open(DOMAIN_RULES_PATH, "r", encoding="utf-8") as f:
    domain_rules_text = f.read()

# ── Load ground truth ──────────────────────────────────────────────────────────
try:
    gt_df = pd.read_csv(GT_CSV, encoding="utf-8")
except UnicodeDecodeError:
    gt_df = pd.read_csv(GT_CSV, encoding="mac_roman")

gt_df.columns = gt_df.columns.str.strip()

def normalize_frame_id(val):
    try:
        return str(int(float(str(val).replace("'", "").strip())))
    except:
        return str(val).replace("'", "").strip()

gt_df["frame_id"] = gt_df["frame_id"].astype(str).apply(normalize_frame_id)

# ── Parse labels from description string ──────────────────────────────────────
SKIP_WELD_STATES = {"Not Welding", "Weld stop", "Weld Stop", "Weld End"}

def parse_labels(description: str) -> dict:
    def extract(pattern, text):
        match = re.search(pattern, str(text), re.IGNORECASE)
        return match.group(1).strip() if match else "N/A"

    labels = {
        "weld_state":            extract(r"Weld State:\s*([^,]+)",            description),
        "arc_stability":         extract(r"Arc Stability:\s*([^,]+)",         description),
        "spatter_particle_size": extract(r"Spatter Particle Size:\s*([^,]+)", description),
        "spatter_volume":        extract(r"Spatter Volume:\s*([^,]+)",        description),
        "melt_pool_length":      extract(r"Melt Pool Length:\s*([^,]+)",      description),
        "melt_pool_stability":   extract(r"Melt Pool Stability:\s*([^,]+)",   description),
        "quality":               extract(r"Quality:\s*([^,]+)",               description),
    }

    # Normalize weld state
    ws = labels["weld_state"]
    if ws in ["Arc Initiation", "Arc Ignition"]:
        labels["weld_state"] = "Arc Ignition"
    elif ws == "Weld stop":
        labels["weld_state"] = "Weld Stop"

    # Normalize quality
    labels["quality"] = labels["quality"].replace("Ê", "").strip().rstrip(".")

    return labels

# ── DSPy Signature — same inputs as manual prompt ─────────────────────────────
class WeldInspection(dspy.Signature):
    """
    You are an expert inspection agent performing visual-metadata-sensor fusion analysis.
    Your output labels will be compared against human expert labels.
    You must match the human labeling schema exactly, including label vocabulary,
    N/A rules, and quality judgment logic.
    """
    domain_rules  = dspy.InputField(desc=(
        "Domain-specific welding inspection rules and thresholds for all 7 categories"
    ))
    metadata      = dspy.InputField(desc=(
        "JSON with melt_pool_region (Area, Width, Height, Orientation, "
        "FeretDiameter0, FeretDiameter90, Cx, Cy) and spatter_features "
        "(spatter_count, total_spatter_area_px, largest_spatter_area_px, mean_spatter_area_px)"
    ))
    thermal_image = dspy.InputField(desc=(
        "Refined thermal overlay image with green melt pool bounding box labeled "
        "'Melt Pool Region', cyan crosshair axes, yellow hotspot marker with X,Y coordinates, "
        "F0/F90/H/Width text annotations, spatter count in top-left corner, "
        "false-color thermal gradient, and colorbar on right"
    ))
    voltage_plot  = dspy.InputField(desc=(
        "Voltage signal plot with 4 rows: raw signal and rolling average, "
        "residual, anomaly events (spikes and extinctions), trend derivative"
    ))
    current_plot  = dspy.InputField(desc=(
        "Current signal plot with 4 rows: raw signal and rolling average, "
        "residual, anomaly events (spikes and extinctions), trend derivative"
    ))

    weld_state            = dspy.OutputField(desc="Arc Ignition | Welding | Not Welding | Weld Stop | Weld End")
    arc_stability         = dspy.OutputField(desc="Stable | Unstable | N/A")
    spatter_particle_size = dspy.OutputField(desc="Fine | Small | Medium | Moderate | Large | N/A")
    spatter_volume        = dspy.OutputField(desc="None | Minimal | Moderate | Extreme")
    melt_pool_length      = dspy.OutputField(desc="Short | Nominal | N/A")
    melt_pool_stability   = dspy.OutputField(desc="Stable | Unstable | N/A")
    quality               = dspy.OutputField(desc="Good | Bad | N/A")

# ── DSPy Modules ───────────────────────────────────────────────────────────────
class ZeroShotAgent(dspy.Module):
    def __init__(self):
        self.inspect = dspy.Predict(WeldInspection)
    def forward(self, thermal_image, metadata, voltage_plot, current_plot, domain_rules):
        return self.inspect(
            thermal_image=thermal_image, metadata=metadata,
            voltage_plot=voltage_plot, current_plot=current_plot,
            domain_rules=domain_rules)

class ZeroShotCoTAgent(dspy.Module):
    def __init__(self):
        self.inspect = dspy.ChainOfThought(WeldInspection)
    def forward(self, thermal_image, metadata, voltage_plot, current_plot, domain_rules):
        return self.inspect(
            thermal_image=thermal_image, metadata=metadata,
            voltage_plot=voltage_plot, current_plot=current_plot,
            domain_rules=domain_rules)

class FewShotAgent(dspy.Module):
    def __init__(self):
        self.inspect = dspy.Predict(WeldInspection)
    def forward(self, thermal_image, metadata, voltage_plot, current_plot, domain_rules):
        return self.inspect(
            thermal_image=thermal_image, metadata=metadata,
            voltage_plot=voltage_plot, current_plot=current_plot,
            domain_rules=domain_rules)

class FewShotCoTAgent(dspy.Module):
    def __init__(self):
        self.inspect = dspy.ChainOfThought(WeldInspection)
    def forward(self, thermal_image, metadata, voltage_plot, current_plot, domain_rules):
        return self.inspect(
            thermal_image=thermal_image, metadata=metadata,
            voltage_plot=voltage_plot, current_plot=current_plot,
            domain_rules=domain_rules)

STRATEGIES = {
    "zero_shot":     (ZeroShotAgent,    0),
    "zero_shot_cot": (ZeroShotCoTAgent, 0),
    "few_shot":      (FewShotAgent,     20),
    "few_shot_cot":  (FewShotCoTAgent,  20),
}

# Build DSPy examples — same inputs as manual prompt 
def build_examples(overlay_dir: Path, metadata_dir: Path,
                   signal_dir: Path, gt_df: pd.DataFrame) -> list:
    examples = []
    for fname in sorted(os.listdir(overlay_dir)):
        if not fname.endswith("_refined_spatter_overlay.png"):
            continue

        frame_id      = fname.replace("_refined_spatter_overlay.png", "")
        thermal_path  = overlay_dir  / fname
        metadata_path = metadata_dir / f"{frame_id}_metadata.json"
        current_path  = signal_dir   / f"{frame_id}_current_signals.png"
        voltage_path  = signal_dir   / f"{frame_id}_voltage_signals.png"

        if not thermal_path.exists():
            print(f"Skipping {frame_id} — overlay image not found")
            continue

        # Match ground truth by last 6 digits
        search_key = frame_id[-6:]
        match = gt_df[gt_df["frame_id"].str.endswith(search_key)]
        if match.empty:
            print(f"Skipping {frame_id} — no ground truth match")
            continue

        labels = parse_labels(str(match.iloc[0].get("description", "")))

        # Skip frames outside vocabulary
        if labels["weld_state"] in SKIP_WELD_STATES:
            print(f"Skipping {frame_id} — weld state '{labels['weld_state']}' outside vocabulary")
            continue

        # Load inputs — same as manual prompt pipeline
        metadata_text = open(metadata_path).read() if metadata_path.exists() else "Not available"
        thermal_img = dspy.Image(str(thermal_path))
        current_img = dspy.Image(str(current_path)) if current_path.exists() else "Not available"
        voltage_img = dspy.Image(str(voltage_path)) if voltage_path.exists() else "Not available"
        example = dspy.Example(
            thermal_image=thermal_img,
            metadata=metadata_text,
            voltage_plot=voltage_img,
            current_plot=current_img,
            domain_rules=domain_rules_text,
            **labels
        ).with_inputs("thermal_image", "metadata", "voltage_plot", "current_plot", "domain_rules")

        examples.append(example)

    return examples

# ── Load datasets ──────────────────────────────────────────────────────────────
print("Loading train dataset...")
dataset = build_examples(TRAIN_OVERLAY, TRAIN_METADATA, TRAIN_SIGNAL, gt_df)
print(f"Train frames loaded: {len(dataset)}")

print("Loading few-shot dataset ...")
few_shot_demos = build_examples(FEW_SHOT_OVERLAY, FEW_SHOT_METADATA, FEW_SHOT_SIGNAL, gt_df)
print(f"Few-shot frames loaded: {len(few_shot_demos)}")

# ── Metric ─────────────────────────────────────────────────────────────────────
def weld_metric(example, pred, trace=None):
    fields = ["weld_state", "arc_stability", "spatter_particle_size",
              "spatter_volume", "melt_pool_length", "melt_pool_stability", "quality"]
    score = sum(
        getattr(example, f, "").strip().lower() == getattr(pred, f, "").strip().lower()
        for f in fields
    ) / len(fields)
    if trace is not None:
        return score >= 0.5
    return score

# ── KFold optimization on train dataset ───────────────────────────────────────
kf = KFold(n_splits=4, shuffle=True, random_state=42)
final_results = {}

for strategy_name, (AgentClass, n_demos) in STRATEGIES.items():
    print(f"\n{'='*60}")
    print(f"STRATEGY: {strategy_name.upper()}")
    print(f"{'='*60}")

    best_score  = -1
    best_agent  = None
    best_fold   = 1
    fold_scores = []

    for fold, (train_idx, dev_idx) in enumerate(kf.split(dataset)):
        trainset = [dataset[i] for i in train_idx]
        devset   = [dataset[i] for i in dev_idx]

        evaluate = Evaluate(
            devset=devset, metric=weld_metric,
            num_threads=2, display_progress=False
        )

        # Baseline on first fold only
        if fold == 0:
            baseline_score = float(evaluate(AgentClass()))
            print(f"Baseline accuracy (fold 1 devset): {baseline_score:.2f}")

        teleprompter = BootstrapFewShot(
            metric=weld_metric,
            max_bootstrapped_demos=n_demos,
            max_rounds=3,
            max_labeled_demos=n_demos
        )

        # For few_shot strategies inject bead 1 demos as seed
        if "few_shot" in strategy_name and n_demos > 0 and few_shot_demos:
            compiled = AgentClass()
            compiled.inspect.demos = few_shot_demos[:n_demos]
            optimized_agent = teleprompter.compile(compiled, trainset=trainset)
        else:
            optimized_agent = teleprompter.compile(AgentClass(), trainset=trainset)

        optimized_score = float(evaluate(optimized_agent))
        fold_scores.append(optimized_score)

        print(f"Fold {fold+1} | Optimized: {optimized_score:.2f} | "
              f"Train: {len(trainset)} | Dev: {len(devset)}")

        if optimized_score > best_score:
            best_score = optimized_score
            best_agent = optimized_agent
            best_fold  = fold + 1

    avg_score = sum(fold_scores) / len(fold_scores)
    print(f"\nBest fold: {best_fold} | Best score: {best_score:.2f} | Avg: {avg_score:.2f}")

    save_path = str(OUTPUT_DIR / f"optimized_{strategy_name}.json")
    best_agent.save(save_path)
    print(f"Saved → {save_path}")

    final_results[strategy_name] = {
        "baseline":  round(baseline_score, 2),
        "best":      round(best_score, 2),
        "avg":       round(avg_score, 2),
        "best_fold": best_fold
    }

# ── Save summary CSV ───────────────────────────────────────────────────────────
summary_df = pd.DataFrame([
    {"strategy": s, **r} for s, r in final_results.items()
])
summary_csv = str(OUTPUT_DIR / "dspy_optimization_summary.csv")
summary_df.to_csv(summary_csv, index=False)

# ── Final summary ──────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("FINAL RESULTS SUMMARY")
print(f"{'='*60}")
print(f"{'Strategy':<20} {'Baseline':>10} {'Best':>8} {'Avg':>8} {'BestFold':>10}")
print(f"{'-'*58}")
for s, r in final_results.items():
    print(f"{s:<20} {r['baseline']:>10.2f} {r['best']:>8.2f} "
          f"{r['avg']:>8.2f} {r['best_fold']:>10}")
print(f"\nOptimized prompts saved to: {OUTPUT_DIR}")
print(f"Summary CSV saved to: {summary_csv}")