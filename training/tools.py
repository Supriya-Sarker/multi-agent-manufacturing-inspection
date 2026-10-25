import os
import pandas as pd
from PIL import Image
import google.generativeai as genai
from langchain_core.tools import tool
from dotenv import load_dotenv
import time
import json
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support
from metrics_logger import record_llm_call

# Mode configuration 
# PROMPT_MODE = "zero_shot"
# PROMPT_MODE = "zero_shot_cot"
# PROMPT_MODE = "few_shot"
PROMPT_MODE = "few_shot_cot"

# PROMPT_MODE = "dspy_zero_shot"
# PROMPT_MODE = "dspy_zero_shot_cot"
# PROMPT_MODE = "dspy_few_shot"
# PROMPT_MODE = "dspy_few_shot_cot"
# EVAL_MODE   = f"{PROMPT_MODE}_no_memory"
EVAL_MODE = f"{PROMPT_MODE}_with_memory"

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Paths 
BASE_DIR    = "/Users/supriya/Documents/data/agentic_system/multi-agent-manufacturing-inspection/training"
BASE_PATH   = Path(BASE_DIR)
PROMPT_BASE = BASE_PATH / "prompts"

# if PROMPT_MODE in ("zero_shot", "few_shot"):
#     PROMPT_PATH = PROMPT_BASE / "vision_prompt_1.txt"
# else:
#     PROMPT_PATH = PROMPT_BASE / "vision_prompt_cot.txt"

if PROMPT_MODE in ("zero_shot", "few_shot"):
    PROMPT_PATH = PROMPT_BASE / "vision_prompt_1.txt"
elif PROMPT_MODE in ("zero_shot_cot", "few_shot_cot"):
    PROMPT_PATH = PROMPT_BASE / "vision_prompt_cot.txt"
elif PROMPT_MODE.startswith("dspy_"):
    strategy = PROMPT_MODE.replace("dspy_", "")
    PROMPT_PATH = PROMPT_BASE / f"vision_prompt_dspy_{strategy}.txt"

DOMAIN_RULES_PATH = PROMPT_BASE / "domain_rules_1.txt"
FEW_SHOT_DIR     = BASE_PATH / "images" / "few_shot"
FEW_SHOT_PROMPTS = PROMPT_BASE / "few_shot"

RESULTS_DIR = BASE_PATH / "results"

# Helpers 
def load_text(path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_prompt_with_domain_rules(prompt_path, domain_rules_path) -> str:
    prompt_text  = load_text(prompt_path)
    domain_rules = load_text(domain_rules_path).strip()
    return prompt_text.replace("{{DOMAIN_RULES}}", domain_rules)


# inspect_waam_image
@tool
def inspect_waam_image(image_path: str) -> str:
    """
    Analyze a WAAM thermal overlay image frame.
    Modes: zero_shot, zero_shot_cot, few_shot, few_shot_cot.
    """
    model      = genai.GenerativeModel("gemini-flash-latest")
    query_base = os.path.basename(image_path).replace("_refined_spatter_overlay.png", "").replace(".png", "")
    query_img  = Image.open(image_path)

    if PROMPT_MODE == "zero_shot":
        prompt_text  = load_prompt_with_domain_rules(PROMPT_PATH, DOMAIN_RULES_PATH)
        query_inputs = [prompt_text, "QUERY THERMAL OVERLAY IMAGE:", query_img]

        start_time = time.perf_counter()
        response   = model.generate_content(query_inputs)
        record_llm_call(eval_mode=EVAL_MODE, frame_id=query_base,
                        call_type="inspect_waam_image_zero_shot",
                        model_name="gemini-flash-latest",
                        start_time=start_time, response=response)
        return response.text

    elif PROMPT_MODE == "zero_shot_cot":
        prompt_text = load_prompt_with_domain_rules(PROMPT_PATH, DOMAIN_RULES_PATH)
        query_inputs = [prompt_text, "QUERY THERMAL OVERLAY IMAGE:", query_img]
        start_time = time.perf_counter()
        response   = model.generate_content(query_inputs)
        record_llm_call(eval_mode=EVAL_MODE, frame_id=query_base,
                        call_type="inspect_waam_image_zero_shot_cot",
                        model_name="gemini-flash-latest",
                        start_time=start_time, response=response)
        return response.text

    elif PROMPT_MODE in ("few_shot", "few_shot_cot"):
        example_ids = sorted([
            f.replace("_refined_spatter_overlay.png", "")
            for f in os.listdir(FEW_SHOT_DIR / "overlay_images")
            if f.endswith("_refined_spatter_overlay.png")
            ])
        contents = []
        for ex_id in example_ids:
            ex_overlay  = FEW_SHOT_DIR / "overlay_images" / f"{ex_id}_refined_spatter_overlay.png"
            ex_metadata = FEW_SHOT_DIR / "metadata"        / f"{ex_id}_metadata.json"
            ex_current  = FEW_SHOT_DIR / "signal_plots"    / f"{ex_id}_current_signals.png"
            ex_voltage  = FEW_SHOT_DIR / "signal_plots"    / f"{ex_id}_voltage_signals.png"
            ex_text     = FEW_SHOT_PROMPTS / f"{ex_id}.txt"

            contents.extend([
                f"EXAMPLE FRAME {ex_id} PROMPT:",      load_text(ex_text),
                f"EXAMPLE FRAME {ex_id} THERMAL OVERLAY IMAGE:", Image.open(ex_overlay),
                f"EXAMPLE FRAME {ex_id} METADATA JSON:",         load_text(ex_metadata),
                f"EXAMPLE FRAME {ex_id} CURRENT SIGNAL PLOT:",   Image.open(ex_current),
                f"EXAMPLE FRAME {ex_id} VOLTAGE SIGNAL PLOT:",   Image.open(ex_voltage),
            ])

        task_text = """
Now analyze the given WAAM thermal frame using the expert-labeled examples as guidance.
Do NOT skip categories. Do NOT hallucinate missing features. Do NOT invent metadata or signal values.

Output must follow EXACT structure:
Weld State:
Arc Stability:
Spatter Particle Size:
Spatter Volume:
Melt Pool Length:
Melt Pool Stability:
Quality:

JUSTIFICATION:
Thermal Visual Evidence:
Metadata Evidence:
Signal Evidence:
Fusion Notes:
Quality Rationale:
"""
        query_inputs = [task_text, "QUERY THERMAL OVERLAY IMAGE:", query_img]
        start_time = time.perf_counter()
        response   = model.generate_content(contents + query_inputs)
        record_llm_call(eval_mode=EVAL_MODE, frame_id=query_base,
                        call_type=f"inspect_waam_image_{PROMPT_MODE}",
                        model_name="gemini-flash-latest",
                        start_time=start_time, response=response)
        return response.text

    else:
        return f"Error: Unknown PROMPT_MODE='{PROMPT_MODE}'"


# inspect_signal_plot
@tool
def inspect_signal_plot(image_path: str) -> str:
    """
    Analyze a voltage or current signal plot image for one inspection frame.
    """
    model = genai.GenerativeModel("gemini-flash-latest")

    try:
        query_img = Image.open(image_path)
        frame_id  = os.path.basename(image_path)\
                      .replace("_current_signals.png", "")\
                      .replace("_voltage_signals.png", "")

        call_type = ("inspect_signal_plot_current"
                     if "current" in image_path.lower()
                     else "inspect_signal_plot_voltage")

        prompt_text = """
Analyze this voltage/current signal plot image.

The plot may contain four rows:
Row 1: Raw signal and rolling average.
Row 2: Residual, computed as raw signal minus rolling average.
Row 3: Anomaly events, including spikes, arc-collapse, short-circuit, or extinction events if present.
Row 4: Smoothed trend derivative, including rising, stable, or falling trend labels if present.

Use only values printed in the plot title, labels, legends, or annotations.
Do not invent spike counts, extinction counts, frame time, or trend labels.
If a value is unreadable, write "unknown".

Return:
Signal type:
Frame time:
Process activity:
Spike count:
Extinction or arc-collapse count:
Trend direction:

Signal-level observation:

Ground-truth support from signal:
Weld State:
Arc Stability:
Spatter Particle Size:
Spatter Volume:
Melt Pool Length:
Melt Pool Stability:
Quality:

Reason:
"""
        start_time = time.perf_counter()
        response   = model.generate_content([prompt_text, query_img])
        record_llm_call(eval_mode=EVAL_MODE, frame_id=frame_id,
                        call_type=call_type,
                        model_name="gemini-flash-latest",
                        start_time=start_time, response=response)
        return response.text

    except Exception as e:
        return f"Error analyzing signal plot: {e}"


# get_human_ground_truth
@tool
def get_human_ground_truth(frame_id: str) -> str:
    """
    Retrieves human expert labels for a given frame ID.
    """
    csv_path = BASE_PATH / "images" / "images_ground_truth.csv"

    print(f"\nDEBUG: Tool triggered for frame {frame_id}")

    try:
        if not csv_path.exists():
            return f"Error: CSV not found at {csv_path}"

        try:
            df = pd.read_csv(csv_path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding="mac_roman")

        # df.columns = df.columns.str.strip()
        # df["frame_id"] = df["frame_id"].astype(str).str.replace("'", "").str.strip()
        df.columns = df.columns.str.strip()

        def normalize_frame_id(val):
            try:
                return str(int(float(val.replace("'", "").strip())))
            except:
                return val.replace("'", "").strip()

        df["frame_id"] = df["frame_id"].astype(str).apply(normalize_frame_id)

        search_key = str(frame_id).replace(".png", "").strip()[-6:]
        match = df[df["frame_id"].str.endswith(search_key)]

        if match.empty:
            msg = f"DATABASE ERROR: No match for suffix {search_key}"
            print(msg)
            return msg

        description   = match["description"].values[0]
        justification = match["justification"].values[0]

        print("\n" + "="*60)
        print(f"VERIFYING DATA ACCESS FOR KEY: {search_key}")
        print(f"HUMAN LABEL: {description}")
        print(f"HUMAN JUSTIFICATION: {justification}")
        print("="*60 + "\n")

        return f"EXPERT_LABEL: {description}\nEXPERT_JUSTIFICATION: {justification}"

    except Exception as e:
        error_msg = f"SYSTEM ERROR IN TOOL: {str(e)}"
        print(error_msg)
        return error_msg


# Score trackers
category_scores_tracker = {
    "WELD STATE":           [],
    "ARC STABILITY":        [],
    "SPATTER PARTICLE SIZE":[],
    "SPATTER VOLUME":       [],
    "MELT POOL LENGTH":     [],
    "MELT POOL STABILITY":  [],
    "QUALITY":              [],
}

evaluation_history = {}


# evaluation_scores_guide 
@tool
def evaluation_scores_guide(
    weld_state:          float,
    arc_stability:       float,
    spatter_size:        float,
    spatter_volume:      float,
    melt_pool_length:    float,
    melt_pool_stability: float,
    quality:             float,
) -> str:
    """
    Saves the binary scores for the current frame into the global aggregator.
    Called once per frame after all seven categories are logged.
    """
    category_scores_tracker["WELD STATE"].append(weld_state)
    category_scores_tracker["ARC STABILITY"].append(arc_stability)
    category_scores_tracker["SPATTER PARTICLE SIZE"].append(spatter_size)
    category_scores_tracker["SPATTER VOLUME"].append(spatter_volume)
    category_scores_tracker["MELT POOL LENGTH"].append(melt_pool_length)
    category_scores_tracker["MELT POOL STABILITY"].append(melt_pool_stability)
    category_scores_tracker["QUALITY"].append(quality)

    return "All 7 category scores successfully stored in the aggregator."


# log_frame_evaluation
@tool
def log_frame_evaluation(
    frame_id:           str,
    category:           str,
    score:              float,
    predicted_label:    str,
    ground_truth_label: str,
) -> str:
    """
    Logs the evaluation result for one frame and one category.
    Requires predicted_label and ground_truth_label for Precision, Recall, F1 computation.
    Uses frame_id + category as a unique key to prevent duplicate logging.
    """
    normalized_category = category.lower().strip().replace(" ", "_")

    category_map = {
        "weld_state":           "weld_state",
        "arc_stability":        "arc_stability",
        "spatter_size":         "spatter_particle_size",
        "spatter_particle_size":"spatter_particle_size",
        "spatter_volume":       "spatter_volume",
        "melt_pool_length":     "melt_pool_length",
        "melt_pool_stability":  "melt_pool_stability",
        "quality":              "quality",
    }

    if normalized_category not in category_map:
        return f"Unknown category: {category}"

    normalized_category = category_map[normalized_category]
    key = (str(frame_id), normalized_category)

    evaluation_history[key] = {
        "frame_id":           str(frame_id),
        "category":           normalized_category,
        "predicted_label":    str(predicted_label).strip(),
        "ground_truth_label": str(ground_truth_label).strip(),
        "score":              float(score),
    }

    return (f"Logged: frame={frame_id} | category={normalized_category} | "
            f"predicted={predicted_label} | ground_truth={ground_truth_label} | score={score}")


# calculate_performance_matrix
@tool
def calculate_performance_matrix() -> str:
    """
    Computes Accuracy, Precision, Recall, F1 per category and overall.
    """
    df = get_full_performance_dataframe()
    if df.empty:
        return "No evaluation history found. Please log frames first."
    return df.to_string(index=False)


def get_full_performance_dataframe() -> pd.DataFrame:
    """
    Computes Accuracy, macro Precision, Recall, F1 per category and overall.
    """
    if not evaluation_history:
        return pd.DataFrame()

    df       = pd.DataFrame(list(evaluation_history.values()))
    categories = df["category"].unique()
    rows     = []

    for cat in sorted(categories):
        cat_data = df[df["category"] == cat]
        y_true   = cat_data["ground_truth_label"].tolist()
        y_pred   = cat_data["predicted_label"].tolist()

        total    = len(cat_data)
        correct  = int((cat_data["score"] == 1.0).sum())
        accuracy = round(correct / total, 4) if total > 0 else 0.0

        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred,
            average="macro",
            zero_division=0
        )

        rows.append({
            "Category":  cat,
            "Total":     total,
            "Correct":   correct,
            "Incorrect": total - correct,
            "Accuracy":  accuracy,
            "Precision": round(float(prec), 4),
            "Recall":    round(float(rec),  4),
            "F1":        round(float(f1),   4),
        })

    # Overall row
    y_true_all = df["ground_truth_label"].tolist()
    y_pred_all = df["predicted_label"].tolist()
    total_all  = len(df)
    correct_all = int((df["score"] == 1.0).sum())

    prec_all, rec_all, f1_all, _ = precision_recall_fscore_support(
        y_true_all, y_pred_all,
        average="macro",
        zero_division=0
    )

    rows.append({
        "Category":  "Overall",
        "Total":     total_all,
        "Correct":   correct_all,
        "Incorrect": total_all - correct_all,
        "Accuracy":  round(correct_all / total_all, 4) if total_all > 0 else 0.0,
        "Precision": round(float(prec_all), 4),
        "Recall":    round(float(rec_all),  4),
        "F1":        round(float(f1_all),   4),
    })

    return pd.DataFrame(rows)


def export_performance_csv(output_dir: str, eval_mode: str) -> str:
    """
    Exports the performance matrix (Accuracy, Precision, Recall, F1)
    to a CSV file labelled with eval_mode.
    """
    df = get_full_performance_dataframe()
    if df.empty:
        print("No evaluation history to export.")
        return ""

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{eval_mode}_performance_metrics.csv")
    df.to_csv(path, index=False)
    print(f"Performance metrics saved → {path}")
    return path