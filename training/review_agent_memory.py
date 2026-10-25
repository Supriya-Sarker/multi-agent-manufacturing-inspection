from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from tools import get_human_ground_truth, evaluation_scores_guide, log_frame_evaluation, calculate_performance_matrix
import json
import os
import time
from pathlib import Path
from metrics_logger import record_llm_call_from_values, record_step_latency
from langchain_core.callbacks import BaseCallbackHandler
from tools import PROMPT_MODE, EVAL_MODE
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)
tools = [get_human_ground_truth, evaluation_scores_guide, log_frame_evaluation, calculate_performance_matrix]

BASE_DIR = Path("/Users/supriya/Documents/data/agentic_system/multi-agent-manufacturing-inspection/training")
DOMAIN_RULES_PATH = BASE_DIR / "prompts" / "domain_rules_1.txt"
REVIEW_SUMMARY_DIR = BASE_DIR / "reviewer_logs" / EVAL_MODE


def load_domain_rules(path: Path = DOMAIN_RULES_PATH) -> str:
    with open(path, "r") as f:
        return f.read().strip()

DOMAIN_RULES = load_domain_rules()

# ── System prompt 
SYSTEM_PROMPT_TEMPLATE = """\
You are a WAAM Quality Auditor. Your task is to compare the Vision Agent Report with the Expert Ground Truth.

First call get_human_ground_truth with frame_id = {{sample_id}}. Use that result as the Expert Ground Truth.

You must review exactly these seven categories:
(i)   Weld State
(ii)  Arc Stability
(iii) Spatter Particle Size
(iv)  Spatter Volume
(v)   Melt Pool Length
(vi)  Melt Pool Stability
(vii) Quality

DOMAIN DECISION RULES
{domain_rules}

STRICT RULE USE POLICY:
Use only the injected Domain Decision Rules above.
Do not invent thresholds, heuristics, modality priorities, or new decision logic.
Do not use general WAAM knowledge unless it is explicitly stated in the Domain Decision Rules.
Do not create a new rule from the Expert Ground Truth.
Use Expert Ground Truth only to determine whether the Vision Agent matched the expected label.
Every critique and fix must be traceable to the injected Domain Decision Rules.

MODALITY POLICY:
Follow the modality priority rules in the Domain Decision Rules.
Do not suggest using signal plots for categories where signal plots cannot determine the label.
Do not suggest using metadata fields that are not listed as valid evidence for that category.
If the rules do not support a specific correction, write:
Fix: Re-check using only the applicable Domain Decision Rules for this category.

FOR EACH CATEGORY:
1. Compare the Vision Agent label/reasoning against the Expert Ground Truth.
2. Decide status:
   Match = same physical label or same core assessment.
   Mismatch = wrong label or contradicts the expert assessment.
   Hallucination = impossible or unsupported claim.
3. Assign score:
   1.0 = Match
   0.0 = Mismatch or Hallucination
4. If score = 0.0, write a fix using only the exact applicable Domain Decision Rule.
   If score = 1.0, write Fix: N/A.

REPORTING FORMAT:
Use plain text only. No Markdown. No asterisks. No backticks.
Keep each category review concise and short.
Critique: maximum 1 sentence.
Fix: maximum 1 sentence.

For each category, use exactly this format:

[CATEGORY NAME]
Status: [Match / Mismatch / Hallucination]
Critique: [Brief semantic comparison. Reference only domain-rule-supported evidence.]
Score: [1.0 / 0.0]
Fix: [Domain-rule-based correction only, or N/A.]

FINAL ACTION:
For EACH of the seven categories, call log_frame_evaluation exactly once with:
frame_id = the current Frame ID from {{sample_id}}
category = the category name
score = the exact numeric score, either 1.0 or 0.0
predicted_label = the label the Vision Agent predicted for this category
ground_truth_label = the label from Expert Ground Truth for this category

Do not call log_frame_evaluation without frame_id.
Do not call log_frame_evaluation more than once for the same frame_id and category.
After all seven categories are logged, call evaluation_scores_guide once with all seven numeric scores.
"""

SYSTEM_PROMPT = SYSTEM_PROMPT_TEMPLATE.format(domain_rules=DOMAIN_RULES)

# ── Prompt & agent ─────────────────────────────────────────────────────────────
# prompt = ChatPromptTemplate.from_messages([
#     ("system", SYSTEM_PROMPT),
#     ("human", "Critique the AI Vision report for Frame {sample_id}:\n\n{vision_report}"),
#     ("placeholder", "{agent_scratchpad}"),
# ])

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("placeholder", "{chat_history}"),
    ("human", "Critique the AI Vision report for Frame {sample_id}:\n\n{vision_report}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
reviewer_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)


# ── Metrics callback ───────────────────────────────────────────────────────────
class ReviewAgentMetricsCallback(BaseCallbackHandler):
    def __init__(self, eval_mode: str, frame_id: str, model_name: str = "gemini-flash-latest"):
        self.eval_mode = eval_mode
        self.frame_id = frame_id
        self.model_name = model_name
        self.llm_start_times = {}

    def on_llm_start(self, serialized, prompts, run_id, **kwargs):
        self.llm_start_times[run_id] = time.perf_counter()

    def on_chat_model_start(self, serialized, messages, run_id, **kwargs):
        self.llm_start_times[run_id] = time.perf_counter()

    def on_llm_end(self, response, run_id, **kwargs):
        start_time = self.llm_start_times.pop(run_id, None)
        latency_sec = time.perf_counter() - start_time if start_time is not None else 0.0

        input_tokens = 0
        output_tokens = 0
        total_tokens = 0

        for generation_list in response.generations:
            for generation in generation_list:
                message = getattr(generation, "message", None)
                usage = getattr(message, "usage_metadata", None)
                if usage is None:
                    usage = getattr(response, "llm_output", {}) or {}
                if usage is None:
                    usage = {}

                input_tokens += int(
                    usage.get("input_tokens", 0)
                    or usage.get("prompt_token_count", 0)
                    or 0
                )
                output_tokens += int(
                    usage.get("output_tokens", 0)
                    or usage.get("candidates_token_count", 0)
                    or 0
                )
                total_tokens += int(
                    usage.get("total_tokens", 0)
                    or usage.get("total_token_count", 0)
                    or 0
                )

        if total_tokens == 0:
            total_tokens = input_tokens + output_tokens

        record_llm_call_from_values(
            eval_mode=self.eval_mode,
            frame_id=self.frame_id,
            call_type="review_agent_llm",
            model_name=self.model_name,
            latency_sec=latency_sec,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens
        )


# Helpers
def save_reviewer_summary(frame_id, summary_text, run_mode: str = "train"):
    save_dir = BASE_DIR / "reviewer_logs" / EVAL_MODE / run_mode
    os.makedirs(save_dir, exist_ok=True)
    file_path = save_dir / f"{frame_id}.json"
    data = {
        "frame_id": frame_id,
        "review_text": summary_text.split("\n")
    }
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# Main entry point 
# def run_review(frame_id: str, ai_report: str, run_mode: str = "train") -> str:
def run_review(frame_id: str, ai_report: str, run_mode: str = "train", chat_history: list = []) -> str:
    print(f"Review Agent is comparing Vision Agent Report with Ground Truth...")

    review_callback = ReviewAgentMetricsCallback(
        eval_mode=EVAL_MODE,   # ← was RAG_EVAL_MODE
        frame_id=frame_id,
        model_name="gemini-flash-latest"
    )

    review_start_time = time.perf_counter()

    response = reviewer_executor.invoke(
        {
            "sample_id": frame_id,
            "vision_report": ai_report,
            "chat_history": chat_history
        },
        config={"callbacks": [review_callback]}
    )

    record_step_latency(
        eval_mode=EVAL_MODE, 
        frame_id=frame_id,
        step_name="review_agent_total",
        start_time=review_start_time
    )

    raw_output = response.get("output", "")
    if isinstance(raw_output, list) and len(raw_output) > 0:
        clean_verdict = raw_output[0].get("text", str(raw_output))
    else:
        clean_verdict = str(raw_output)

    clean_verdict = clean_verdict.replace("\\n", "\n").strip()

    save_reviewer_summary(frame_id, clean_verdict, run_mode)
    return clean_verdict


def print_final_matrix():
    print("\n" + "="*50)
    print("CALCULATING FINAL PERFORMANCE MATRIX")
    print("="*50)
    matrix_output = calculate_performance_matrix.invoke({})
    print(matrix_output)