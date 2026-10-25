from __future__ import annotations
import re
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from vision_agent_memory import analyze_waam_frame
from review_agent_memory import run_review

MAX_RETRIES = 2  # how many times vision agent can retry per frame

# State schema
class WAAMInspectionState(TypedDict):
    frame_id: str
    image_path: str
    metadata_path: str
    signal_plot_dir: str
    vision_report: str
    review_verdict: str
    category_scores: Dict[str, float]
    violations: List[str]
    prev_violations: List[str]
    focus_areas: List[str]
    chat_history: List
    retry_count: int
    should_retry: bool
    route: List[str]


# Parser 
def parse_review_verdict(verdict_text: str):
    """
    Parses the review agent's plain-text output to extract:
    - category_scores: dict of {category: score}
    - violations: list of categories scoring < 1.0
    - focus_areas: list of fix instructions for the vision agent to focus on
    """
    categories = [
        "Weld State", "Arc Stability", "Spatter Particle Size",
        "Spatter Volume", "Melt Pool Length", "Melt Pool Stability",
        "Quality"
        ]

    category_scores = {}
    violations = []
    focus_areas = []

    for category in categories:
        # Find score after each category block
        score_pattern = rf"{re.escape(category)}.*?Score:\s*([\d.]+)"
        score_match = re.search(score_pattern, verdict_text, re.DOTALL | re.IGNORECASE)

        if score_match:
            score = float(score_match.group(1))
            category_scores[category] = score

            if score < 1.0:
                violations.append(category)

                # Extract the Fix line for that category to use as focus
                fix_pattern = rf"{re.escape(category)}.*?Fix:\s*(.+?)(?=\n\[|\Z)"
                fix_match = re.search(fix_pattern, verdict_text, re.DOTALL | re.IGNORECASE)

                if fix_match:
                    fix_text = fix_match.group(1).strip().split("\n")[0].strip()
                    if fix_text and fix_text.upper() != "N/A":
                        focus_areas.append(f"{category}: {fix_text}")

    return category_scores, violations, focus_areas


# Nodes 
def vision_node(state: WAAMInspectionState) -> dict:
    retry = state.get("retry_count", 0)
    route = state.get("route", [])
    route = route + [f"vision_node (attempt #{retry + 1})"]

    print(f"\n→ [vision_node] attempt #{retry + 1} | frame: {state['frame_id']}")

    from pathlib import Path
    vision_report = analyze_waam_frame(
        frame_id=state["frame_id"],
        image_path=state["image_path"],
        metadata_path=state["metadata_path"],
        focus_areas=state.get("focus_areas", []),
        chat_history=state.get("chat_history", []),
        signal_plot_dir=Path(state["signal_plot_dir"]) if state.get("signal_plot_dir") else None
    )

    return {"vision_report": vision_report, "route": route}


def review_node(state: WAAMInspectionState) -> dict:
    route = state.get("route", [])
    retry_count = state.get("retry_count", 0)
    prev_violations = state.get("violations", [])

    print(f"\n→ [review_node] reviewing frame: {state['frame_id']}")

    verdict = run_review(
        state["frame_id"],
        state["vision_report"],
        run_mode="train",
        chat_history=state.get("chat_history", [])
    )
    category_scores, violations, focus_areas = parse_review_verdict(verdict)

    # Update chat history with this attempt
    from langchain_core.messages import HumanMessage, AIMessage
    updated_history = list(state.get("chat_history", []))
    updated_history.append(HumanMessage(content=state["vision_report"]))
    updated_history.append(AIMessage(content=verdict))

    # Stop if violations did not reduce
    violation_count_reduced = len(violations) < len(prev_violations) if retry_count > 0 else True
    should_retry = len(violations) > 0 and retry_count < MAX_RETRIES and violation_count_reduced

    route = route + [
        f"review_node (attempt #{retry_count + 1}) → "
        f"{'retry' if should_retry else 'done'} | violations: {violations}"
    ]

    print(f"→ [review_node] violations: {violations} | should_retry: {should_retry}")

    return {
        "review_verdict": verdict,
        "category_scores": category_scores,
        "violations": violations,
        "prev_violations": prev_violations,
        "focus_areas": focus_areas,
        "chat_history": updated_history,
        "retry_count": retry_count + 1,
        "should_retry": should_retry,
        "route": route
    }


# Router 
def should_continue(state: WAAMInspectionState) -> str:
    if state.get("should_retry", False):
        return "retry"
    return "done"

def build_inspection_graph():
    memory = MemorySaver()
    graph = StateGraph(WAAMInspectionState)
    graph.add_node("vision", vision_node)
    graph.add_node("review", review_node)
    graph.set_entry_point("vision")
    graph.add_edge("vision", "review")
    graph.add_conditional_edges(
        "review",
        should_continue,
        {"retry": "vision", "done": END}
    )
    return graph.compile(checkpointer=memory)

inspection_graph = build_inspection_graph()