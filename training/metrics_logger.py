import time
import pandas as pd

# Update these from the official Gemini pricing page for your exact model.
# Prices are USD per 1 million tokens.
MODEL_PRICING = {
    "gemini-flash-latest": {
        "input_per_1m": 1.50,   # Gemini 3.5 Flash rate
        "output_per_1m": 9.00,  # Gemini 3.5 Flash rate
    }
}
# MODEL_PRICING = {
#     "gemini-2.5-flash": {
#         "input_per_1m": 0.30,
#         "output_per_1m": 2.50,
#     }
# }

llm_call_records = []
step_latency_records = []


def reset_metrics():
    llm_call_records.clear()
    step_latency_records.clear()


def _get_usage_value(usage, field_name):
    if usage is None:
        return 0

    if hasattr(usage, field_name):
        value = getattr(usage, field_name)
        return int(value) if value is not None else 0

    if isinstance(usage, dict):
        value = usage.get(field_name, 0)
        return int(value) if value is not None else 0

    return 0


def record_llm_call(
    eval_mode: str,
    frame_id: str,
    call_type: str,
    model_name: str,
    start_time: float,
    response
):
    latency_sec = time.perf_counter() - start_time
    usage = getattr(response, "usage_metadata", None)

    input_tokens = _get_usage_value(usage, "prompt_token_count")
    output_tokens = _get_usage_value(usage, "candidates_token_count")
    total_tokens = _get_usage_value(usage, "total_token_count")

    pricing = MODEL_PRICING.get(model_name, {"input_per_1m": 0.0, "output_per_1m": 0.0})

    input_cost = (input_tokens / 1_000_000) * pricing["input_per_1m"]
    output_cost = (output_tokens / 1_000_000) * pricing["output_per_1m"]
    total_cost = input_cost + output_cost

    llm_call_records.append({
        "eval_mode": eval_mode,
        "frame_id": frame_id,
        "call_type": call_type,
        "model_name": model_name,
        "latency_sec": latency_sec,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "input_cost_usd": input_cost,
        "output_cost_usd": output_cost,
        "total_cost_usd": total_cost,
    })

def record_llm_call_from_values(
    eval_mode: str,
    frame_id: str,
    call_type: str,
    model_name: str,
    latency_sec: float,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int = None
):
    if total_tokens is None:
        total_tokens = input_tokens + output_tokens

    pricing = MODEL_PRICING.get(
        model_name,
        {"input_per_1m": 0.0, "output_per_1m": 0.0}
    )

    input_cost = (input_tokens / 1_000_000) * pricing["input_per_1m"]
    output_cost = (output_tokens / 1_000_000) * pricing["output_per_1m"]
    total_cost = input_cost + output_cost

    llm_call_records.append({
        "eval_mode": eval_mode,
        "frame_id": frame_id,
        "call_type": call_type,
        "model_name": model_name,
        "latency_sec": latency_sec,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "input_cost_usd": input_cost,
        "output_cost_usd": output_cost,
        "total_cost_usd": total_cost,
    })

def record_step_latency(eval_mode: str, frame_id: str, step_name: str, start_time: float):
    latency_sec = time.perf_counter() - start_time

    step_latency_records.append({
        "eval_mode": eval_mode,
        "frame_id": frame_id,
        "step_name": step_name,
        "latency_sec": latency_sec,
    })


def summarize_metrics(eval_mode: str, test_frame_count: int, total_runtime_sec: float):
    df = pd.DataFrame(llm_call_records)

    if df.empty:
        input_tokens = output_tokens = total_tokens = 0
        total_cost = 0.0
        total_llm_latency = 0.0
    else:
        input_tokens = int(df["input_tokens"].sum())
        output_tokens = int(df["output_tokens"].sum())
        total_tokens = int(df["total_tokens"].sum())
        total_cost = float(df["total_cost_usd"].sum())
        total_llm_latency = float(df["latency_sec"].sum())

    avg_latency_per_frame = total_runtime_sec / test_frame_count if test_frame_count > 0 else 0.0
    cost_per_frame = total_cost / test_frame_count if test_frame_count > 0 else 0.0
    tokens_per_frame = total_tokens / test_frame_count if test_frame_count > 0 else 0.0

    return {
        "eval_mode": eval_mode,
        "test_frames": test_frame_count,
        "avg_latency_per_frame_sec": round(avg_latency_per_frame, 3),
        "total_runtime_sec": round(total_runtime_sec, 3),
        "total_llm_latency_sec": round(total_llm_latency, 3),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "total_cost_usd": round(total_cost, 6),
        "cost_per_frame_usd": round(cost_per_frame, 6),
        "tokens_per_frame": round(tokens_per_frame, 2),
    }


def estimate_scaling(summary: dict, frame_counts=(36, 100, 1000, 10000)):
    rows = []

    for n in frame_counts:
        rows.append({
            "frames": n,
            "estimated_runtime_sec": round(summary["avg_latency_per_frame_sec"] * n, 3),
            "estimated_tokens": round(summary["tokens_per_frame"] * n),
            "estimated_cost_usd": round(summary["cost_per_frame_usd"] * n, 6),
        })

    return pd.DataFrame(rows)


def export_metrics_csv(output_dir: str, eval_mode: str):
    llm_df = pd.DataFrame(llm_call_records)
    step_df = pd.DataFrame(step_latency_records)

    llm_path = f"{output_dir}/{eval_mode}_llm_call_metrics.csv"
    step_path = f"{output_dir}/{eval_mode}_step_latency_metrics.csv"

    llm_df.to_csv(llm_path, index=False)
    step_df.to_csv(step_path, index=False)

    return llm_path, step_path