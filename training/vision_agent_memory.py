from __future__ import annotations
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from tools_memory import inspect_waam_image, inspect_signal_plot
from tools_memory import PROMPT_MODE, EVAL_MODE
import json
from pathlib import Path
import time
from metrics_logger import record_step_latency

load_dotenv()

BASE_DIR = Path("/Users/supriya/Documents/data/agentic_system/multi-agent-manufacturing-inspection/training")
# SIGNAL_PLOT_DIR = BASE_DIR / "images" / "signal_plots_combined_images"
SIGNAL_PLOT_DIR = None
if PROMPT_MODE in ("zero_shot", "few_shot"):
    PROMPT_PATH = BASE_DIR / "prompts" / "vision_prompt_1.txt"
else:
    PROMPT_PATH = BASE_DIR / "prompts" / "vision_prompt_cot.txt"

DOMAIN_RULES_PATH = BASE_DIR / "prompts" / "domain_rules_1.txt"

def find_signal_plot_paths(frame_id: str, signal_plot_dir: Path = None):
    """
    Finds voltage and current signal plot images for a given frame ID.
    """
    if signal_plot_dir is None or not signal_plot_dir.exists():
        return None, None
    signal_plot_dir = Path(signal_plot_dir)

    search_key = str(frame_id).replace(".png", "").strip()[-6:]
    voltage_plot_path = None
    current_plot_path = None

    for plot_path in signal_plot_dir.iterdir():
        if plot_path.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
            continue
        file_name = plot_path.name.lower()
        if search_key in file_name and "voltage" in file_name:
            voltage_plot_path = plot_path
        if search_key in file_name and "current" in file_name:
            current_plot_path = plot_path

    return voltage_plot_path, current_plot_path


llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

tools = [inspect_waam_image, inspect_signal_plot]


def load_prompt(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_domain_rules(path: Path = DOMAIN_RULES_PATH) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

system_prompt_text = load_prompt(PROMPT_PATH)
domain_rules = load_domain_rules()
# system_prompt_text = system_prompt_text.replace("{domain_rules}", domain_rules)
system_prompt_text = system_prompt_text.replace("{{DOMAIN_RULES}}", domain_rules)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt_text),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
vision_agent = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=False,
    handle_parsing_errors=True
)


def load_metadata_text(metadata_path: str, max_chars: int = 6000) -> str:
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        metadata_text = json.dumps(metadata, indent=2)
        if len(metadata_text) > max_chars:
            metadata_text = metadata_text[:max_chars] + "\n... [metadata truncated]"
        return metadata_text
    except Exception as e:
        return f"Error loading metadata: {e}"

def analyze_waam_frame(frame_id: str, image_path: str, metadata_path: str,
                       focus_areas: list = [],
                       chat_history: list = [],
                       signal_plot_dir: Path = None) -> str:
    """
    Analyzes the WAAM overlay image, metadata, and available signal plots.
    """
    print("Vision Agent is analyzing overlay image and metadata\n")

    metadata_text = load_metadata_text(metadata_path)
    voltage_plot_path, current_plot_path = find_signal_plot_paths(frame_id, signal_plot_dir)
    focus_block = ""
    if focus_areas:
        focus_lines = "\n".join(f"- {f}" for f in focus_areas)
        focus_block = f"\nPrevious attempt was flagged. Pay close attention to:\n{focus_lines}\n"

    query = f"""
        Analyze this inspection frame using the available inputs.
        {focus_block}
        Frame ID: {frame_id}
        Thermal overlay image path:
        {image_path}
        Metadata JSON path:
        {metadata_path}
        Metadata content:
        {metadata_text}
        Voltage signal plot:
        {voltage_plot_path if voltage_plot_path else "not available"}
        Current signal plot:
        {current_plot_path if current_plot_path else "not available"}

Mandatory Tool-use instruction:
1. Call inspect_waam_image on the thermal overlay image path.
2. If the voltage signal plot path is available, you must call inspect_signal_plot on the voltage signal plot path.
3. If the current signal plot path is available, you must call inspect_signal_plot on the current signal plot path.

Decision instruction:
Use the thermal overlay image and metadata mainly for:
- Spatter Particle Size
- Spatter Volume
- Melt Pool Length
- Melt Pool Stability
- Quality

Use voltage/current signal plots mainly for:
- Arc Stability
- Weld State
- signal-based support for Quality

Do not infer spatter particle size or melt pool length from signal plots alone.
If evidence is weak, use the valid closest label and explain uncertainty only in the justification.
"""

    start_time = time.perf_counter()
    response = vision_agent.invoke({"input": query, "chat_history": chat_history})

    record_step_latency(
        eval_mode=EVAL_MODE,
        frame_id=frame_id,
        step_name="vision_agent_total",
        start_time=start_time
    )

    raw_output = response.get("output")

    try:
        if isinstance(raw_output, list) and len(raw_output) > 0:
            clean_text = raw_output[0].get("text", str(raw_output))
        elif isinstance(raw_output, dict):
            clean_text = raw_output.get("text", str(raw_output))
        else:
            clean_text = str(raw_output)

        clean_text = clean_text.replace("\\n", "\n").replace("\n\n", "\n")
        return clean_text.strip()

    except Exception as e:
        return f"Error parsing report: {e}"