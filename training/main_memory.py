import os
from graph import inspection_graph
from review_agent_memory import print_final_matrix
from tools_memory import (
    category_scores_tracker,
    evaluation_history,
    PROMPT_MODE,
    EVAL_MODE,
    export_performance_csv
)
import time
from metrics_logger import reset_metrics, summarize_metrics, estimate_scaling, export_metrics_csv
from pathlib import Path
import pandas as pd
import signal
import json

# BASE_DIR = "/Users/supriya/Documents/data/agentic_system/multi-agent-manufacturing-inspection/training/images"
BASE_DIR = Path("/Users/supriya/Documents/data/agentic_system/multi-agent-manufacturing-inspection/training/images")
RUN_MODE = "train"   # switch to "test" for test run
# RUN_MODE = "test" 
if RUN_MODE == "train":
    # BEAD_IDS = ["bead_2", "bead_3", "bead_4"]
    DATASET_FOLDER = BASE_DIR / "train_dataset"
else:
    # BEAD_IDS = ["bead_5", "bead_6"]
    DATASET_FOLDER = BASE_DIR / "test_dataset"

# OUTPUT_DIR = "/Users/supriya/Documents/data/agentic_system/multi-agent-manufacturing-inspection/training/results"
OUTPUT_DIR = os.path.join(
    "/Users/supriya/Documents/data/agentic_system/multi-agent-manufacturing-inspection/training/results",
    RUN_MODE
)
OVERLAY_DIR     = DATASET_FOLDER / "overlay_images"
METADATA_DIR    = DATASET_FOLDER / "metadata"
SIGNAL_PLOT_DIR = DATASET_FOLDER / "signal_plots"

all_frames = []

if not OVERLAY_DIR.exists():
    print(f"Warning: overlay folder not found: {OVERLAY_DIR}")
else:
    for fname in sorted(os.listdir(OVERLAY_DIR)):
        if not fname.endswith("_refined_spatter_overlay.png"):
            continue

        frame_id      = fname.replace("_refined_spatter_overlay.png", "")
        image_path    = OVERLAY_DIR / fname
        metadata_path = METADATA_DIR / f"{frame_id}_metadata.json"

        if not metadata_path.exists():
            print(f"Warning: metadata not found for frame {frame_id}")
            continue

        all_frames.append({
            "frame_id":      frame_id,
            "image_path":    str(image_path),
            "metadata_path": str(metadata_path),
        })

print(f"Run mode:          {RUN_MODE}")
print(f"Eval mode:         {EVAL_MODE}")
print(f"Prompt mode:       {PROMPT_MODE}")
print(f"Total frames:      {len(all_frames)}")

valid_modes = ["zero_shot", "zero_shot_cot", "few_shot", "few_shot_cot"]
if PROMPT_MODE not in valid_modes:
    raise ValueError(f"Unknown PROMPT_MODE: {PROMPT_MODE}")

for key in category_scores_tracker:
    category_scores_tracker[key] = []
evaluation_history.clear()
reset_metrics()
run_start_time = time.perf_counter()

def timeout_handler(signum, frame):
    raise TimeoutError("Frame processing exceeded the allowed time")

signal.signal(signal.SIGALRM, timeout_handler)
processed_frames = 0
successful_runtime_sec = 0.0

# for i, frame in enumerate(all_frames, 1):
for i, frame in enumerate(all_frames, 1):
    frame_id      = frame["frame_id"]
    image_path    = frame["image_path"]
    metadata_path = frame["metadata_path"]

    print("\n" + "="*60)
    print(f"FRAME [{i}/{len(all_frames)}]  ID: {frame_id}")
    print("="*60)

    try:
        frame_start_time = time.perf_counter()
        # Maximum time allowed for this entire frame
        signal.alarm(300)   # 300 seconds = 5 minutes

        final_state = inspection_graph.invoke(
            {
                "frame_id":      frame_id,
                "image_path":    image_path,
                "metadata_path": metadata_path,
                "signal_plot_dir": str(SIGNAL_PLOT_DIR),
                "vision_report": "",
                "review_verdict": "",
                "category_scores": {},
                "violations": [],
                "prev_violations": [],
                "focus_areas": [],
                "chat_history": [],
                "retry_count": 0,
                "should_retry": False,
                "route": []
            },
            config={"configurable": {"thread_id": frame_id}}
        )

        print("\nVISION AGENT REPORT:\n" + final_state["vision_report"])
        print("\nREVIEW AGENT VERDICT:\n" + final_state["review_verdict"])

        print("\n--- Decision Route ---")
        for step in final_state["route"]:
            print("  " + step)

        # Save frame state for visualization
        VIZ_DIR = os.path.join(
            "/Users/supriya/Documents/data/agentic_system/multi-agent-manufacturing-inspection/training/results",
            RUN_MODE, "memory_states"
        )
        os.makedirs(VIZ_DIR, exist_ok=True)
        viz_data = {
            "frame_id":        frame_id,
            "route":           final_state.get("route", []),
            "category_scores": final_state.get("category_scores", {}),
            "violations":      final_state.get("violations", []),
            "retry_count":     final_state.get("retry_count", 0),
            "focus_areas":     final_state.get("focus_areas", []),
        }
        with open(os.path.join(VIZ_DIR, f"{frame_id}_state.json"), "w") as f:
            json.dump(viz_data, f, indent=2)

        # Count only successfully completed frames
        processed_frames += 1

        # Add runtime only for successfully completed frames
        successful_runtime_sec += time.perf_counter() - frame_start_time

    except TimeoutError:
        print(f"\nTIMEOUT: Frame {frame_id} took longer than 5 minutes.")
        print("Skipping this frame and continuing to the next frame...")

    except Exception as e:
        print(f"Error processing frame {frame_id}: {e}")
        print("Skipping this frame and continuing...")

    finally:
        # Cancel the timer before starting the next frame
        signal.alarm(0)

print(f"Total scheduled frames: {len(all_frames)}")
print(f"Successfully processed frames: {processed_frames}")
print(f"Skipped frames: {len(all_frames) - processed_frames}")

print("\n" + "="*40)
print("FINAL AGGREGATED ACCURACY")
print("="*40)
print_final_matrix()

# total_runtime_sec = time.perf_counter() - run_start_time
actual_wall_clock_sec = time.perf_counter() - run_start_time
total_runtime_sec = successful_runtime_sec

print(f"Successful processing runtime: {successful_runtime_sec:.2f} sec")
print(f"Actual wall-clock runtime: {actual_wall_clock_sec:.2f} sec")

# metrics_summary = summarize_metrics(
#     eval_mode=f"{EVAL_MODE}_{RUN_MODE}",
#     test_frame_count=processed_frames,
#     total_runtime_sec=total_runtime_sec
# )

metrics_summary = summarize_metrics(
    eval_mode=f"{EVAL_MODE}_{RUN_MODE}",
    test_frame_count=processed_frames,
    total_runtime_sec=total_runtime_sec
)

print("\n" + "="*40)
print("COST / LATENCY / SCALING METRICS")
print("="*40)
for key, value in metrics_summary.items():
    print(f"{key}: {value}")

scaling_df = estimate_scaling(
    metrics_summary,
    frame_counts=(60, 100)
)
print("\nSCALING ESTIMATES")
print(scaling_df.to_string(index=False))

os.makedirs(OUTPUT_DIR, exist_ok=True)

llm_metrics_path, step_metrics_path = export_metrics_csv(
    output_dir=OUTPUT_DIR,
    eval_mode=f"{EVAL_MODE}_{RUN_MODE}"
)
performance_path = export_performance_csv(
    output_dir=OUTPUT_DIR,
    eval_mode=f"{EVAL_MODE}_{RUN_MODE}"
)

print("\nSaved LLM call metrics to:     " + llm_metrics_path)
print("Saved step latency metrics to:  " + step_metrics_path)
print("Saved performance metrics to:   " + performance_path)

summary_path = os.path.join(
    OUTPUT_DIR,
    f"{EVAL_MODE}_{RUN_MODE}_summary.csv"
)
summary_df = pd.DataFrame([{
    "eval_mode":             f"{EVAL_MODE}_{RUN_MODE}",
    "prompt_mode":           PROMPT_MODE,
    "run_mode":              RUN_MODE,
    "total_frames":          metrics_summary.get("total_frames", 0),
    "total_runtime_sec":     metrics_summary.get("total_runtime_sec", 0),
    "avg_latency_per_frame": metrics_summary.get("avg_latency_per_frame", 0),
    "total_cost_usd":        metrics_summary.get("total_cost_usd", 0),
    "avg_cost_per_frame":    metrics_summary.get("avg_cost_per_frame", 0),
    "total_input_tokens":    metrics_summary.get("total_input_tokens", 0),
    "total_output_tokens":   metrics_summary.get("total_output_tokens", 0),
    "total_tokens":          metrics_summary.get("total_tokens", 0),
}])
summary_df.to_csv(summary_path, index=False)
print(f"Summary metrics saved → {summary_path}")