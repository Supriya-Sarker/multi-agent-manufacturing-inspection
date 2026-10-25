"""
Renders the WAAM Inspection Decision Flow visualization directly from your
memory_states/{frame_id}_state.json files — no separate merge step needed.

This is the same visual style/logic as your visualize_memory.py, just reading
straight from the per-frame files main_memory.py writes instead of a merged
run_memory.json.

Usage:
    python plot_memory_states.py
"""

import json
import os
from pathlib import Path

# ── Config — EDIT THESE ──────────────────────────────────────────────
MEMORY_STATES_DIR = "/Users/supriya/Documents/data/agentic_system/multi-agent-manufacturing-inspection/training/results/train/memory_states"
OUTPUT_HTML = "/Users/supriya/Documents/data/agentic_system/multi-agent-manufacturing-inspection/training/results/memory_visual.html"
OUTPUT_PNG  = "/Users/supriya/Documents/data/agentic_system/multi-agent-manufacturing-inspection/training/results/memory_visual.png"

# Pick exactly which frames to plot, in the order you want them to appear.
# Leave as an empty list [] to plot every frame found in MEMORY_STATES_DIR instead.
FRAME_IDS = [
    "1763658143019336",
    "1763658148154901",
    "1763658156714201",
]


def load_all_frames(memory_states_dir: str, frame_ids: list = None) -> dict:
    """
    Reads {frame_id}_state.json files into {frame_id: {...}}.
    If frame_ids is given (non-empty), only those frames are loaded, in that
    exact order. Otherwise every *_state.json file in the folder is loaded.
    """
    memory_states_dir = Path(memory_states_dir)
    if not memory_states_dir.exists():
        raise FileNotFoundError(f"memory_states folder not found: {memory_states_dir}")

    frames = {}

    if frame_ids:
        missing = []
        for fid in frame_ids:
            fpath = memory_states_dir / f"{fid}_state.json"
            if not fpath.exists():
                missing.append(fid)
                continue
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            frames[fid] = {
                "route":           data.get("route", []),
                "category_scores": data.get("category_scores", {}),
                "violations":      data.get("violations", []),
                "retry_count":     data.get("retry_count", 0),
            }
        if missing:
            print(f"WARNING: {len(missing)} frame_id(s) not found in {memory_states_dir}: {missing}")
    else:
        for fpath in sorted(memory_states_dir.glob("*_state.json")):
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            frame_id = str(data.get("frame_id", fpath.stem.replace("_state", "")))
            frames[frame_id] = {
                "route":           data.get("route", []),
                "category_scores": data.get("category_scores", {}),
                "violations":      data.get("violations", []),
                "retry_count":     data.get("retry_count", 0),
            }

    return frames


all_memory = load_all_frames(MEMORY_STATES_DIR, FRAME_IDS)
frames_json = json.dumps(all_memory)

html = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>WAAM Decision Flow</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: Arial, sans-serif;
  background: #ffffff;
  color: #000000;
  padding: 32px;
}
h1 { color: #7eb8f7; font-size: 16px; margin-bottom: 4px; }
.sub {
  color: #444;
  font-size: 11px;
  margin-bottom: 32px;
}

/* ── Timeline container ── */
.timeline { display: flex; flex-direction: column; gap: 40px; }

/* ── Frame block ── */
.frame-block { display: flex; flex-direction: column; gap: 56px; }
.frame-label {
  font-size: 12px; color: #7eb8f7;
  margin-bottom: 12px;
  display: flex; align-items: center; gap: 10px;
}
.frame-label .fid { color: #555; font-size: 11px; }

/* ── Attempt row ── */
.attempt-row {
  display: flex;
  align-items: flex-start;
  gap: 0;
  position: relative;
}

/* connector line between rows */
.attempt-row + .attempt-row::before {
  content: '';
  position: absolute;
  left: 92px;
  top: -36px;
  width: 2px;
  height: 36px;
  background: #c0c0c0;
}

/* ── Attempt index ── */
.attempt-num {
  width: 84px;
  flex-shrink: 0;
  padding-top: 10px;
  font-size: 11px;
  color: #333;
  text-align: right;
  padding-right: 10px;
}

/* ── Cards row ── */
.cards {
  display: flex;
  align-items: flex-start;
  gap: 0;
  flex: 1;
}

/* ── Node column: box + JSON-style text beneath it ── */
.node-col {
  display: flex;
  flex-direction: column;
}
.below-text {
  font-size: 9.5px;
  font-family: "Courier New", monospace;
  margin-top: 6px;
  max-width: 260px;
  line-height: 1.5;
}
.below-text.focus-text { color: #9a8a00; }
.below-text.viol-text  { color: #c76b00; }
.below-text.done-text  { color: #146b38; font-weight: bold; font-family: Arial, sans-serif; font-size: 10.5px; }

/* ── Arrow ── */
.arr-wrap {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 0 6px;
  padding-top: 6px;
}
.arr-label {
  font-size: 8.5px;
  color: #666;
  margin-bottom: 1px;
  white-space: nowrap;
}
.arr {
  color: #3a3d4a;
  font-size: 18px;
}

/* ── Node card base ── */
.node {
  border-radius: 6px;
  padding: 8px 12px;
  font-size: 11px;
  min-width: 120px;
  max-width: 260px;
  border: 1.5px solid #111111;
}
.node .node-title {
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 1px;
  margin-bottom: 4px;
  opacity: 0.7;
}
.node .node-body { line-height: 1.5; }

/* Vision node */
.node-vision {
  background: #eaf3ff;
  border-color: #111111;
  color: #123a63;
}

/* Review retry */
.node-retry {
  background: #fff0e6;
  border-color: #111111;
  color: #8a3f00;
}
/* Review done clean */
.node-done-clean {
  background: #e9f9ef;
  border-color: #111111;
  color: #146b38;
}
/* Review done violations */
.node-done-viol {
  background: #ffecec;
  border-color: #111111;
  color: #8b1e1e;
}

/* ── Score strip ── */
.score-strip {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  margin-top: 5px;
}
.sc {
  font-size: 9px;
  padding: 1px 5px;
  border-radius: 3px;
  white-space: nowrap;
}
.sc-ok {
  background: #e9f9ef;
  color: #146b38;
  border: 1px solid #111111;
}

.sc-bad {
  background: #ffecec;
  color: #8b1e1e;
  border: 1px solid #111111;
}

.sc-mid {
  background: #fff0e6;
  color: #8a3f00;
  border: 1px solid #111111;
}
/* ── Summary bar ── */
.summary {
  display: flex; gap: 12px; margin-bottom: 28px; flex-wrap: wrap;
}
.scard {
  background: #ffffff;
  border: 1.5px solid #111111;
  border-radius: 6px;
  padding: 10px 16px;
}
.scard .sl { font-size: 10px; color: #666; margin-bottom: 2px; }
.scard .sv { font-size: 20px; font-weight: bold; color: #7eb8f7; }

.divider {
  height: 1px;
  background: #d0d0d0;
  margin: 8px 0 0 84px;
}
</style>
</head>
<body>
<h1>WAAM Inspection — Decision Flow</h1>
<p class="sub">State · Focus Areas · Decisions per attempt · Generated from memory_states/*.json</p>

<div class="summary" id="summary"></div>
<div class="timeline" id="timeline"></div>
<svg id="connectorSvg" style="position:absolute; top:0; left:0; pointer-events:none; overflow:visible;"></svg>

<script>
const CATS = ["Weld State","Arc Stability","Spatter Particle Size","Spatter Volume",
              "Melt Pool Length","Melt Pool Stability","Quality"];
const CAT_SHORT = {"Weld State":"WS","Arc Stability":"ARC","Spatter Particle Size":"SPS",
  "Spatter Volume":"SV","Melt Pool Length":"MPL","Melt Pool Stability":"MPS",
  "Quality":"Q"};

const data = """ + frames_json + r""";
const frames = Object.entries(data);

// Summary
const totalFrames = frames.length;
const totalRetries = frames.reduce((s,[,f])=>s+(f.retry_count||0),0);
const clean = frames.filter(([,f])=>(f.violations||[]).length===0).length;
document.getElementById("summary").innerHTML = `
  <div class="scard"><div class="sl">Frames</div><div class="sv">${totalFrames}</div></div>
  <div class="scard"><div class="sl">Avg Attempts</div><div class="sv">${totalFrames?(1+totalRetries/totalFrames).toFixed(1):0}</div></div>
  <div class="scard"><div class="sl">Clean Frames</div><div class="sv" style="color:#60f0a0">${clean}</div></div>
  <div class="scard"><div class="sl">With Violations</div><div class="sv" style="color:#f06060">${totalFrames-clean}</div></div>
`;

const timeline = document.getElementById("timeline");
const retryConnectors = [];  // { from: reviewNodeId, to: nextVisionNodeId }

frames.forEach(([frameId, frame], fi) => {
  const route = frame.route || [];
  const finalViolations = frame.violations || [];
  const finalScores = frame.category_scores || {};
  const retries = frame.retry_count || 0;

  // Parse route into attempts
  const attempts = [];
  let cur = null;
  route.forEach(step => {
    if (step.includes("vision_node")) {
      cur = { vision: step, review: null };
      attempts.push(cur);
    } else if (step.includes("review_node") && cur) {
      cur.review = step;
    }
  });

  const focusPerAttempt = [];
  attempts.forEach((att, i) => {
    if (i === 0) {
      focusPerAttempt.push([]);
    } else {
      const prevReview = attempts[i-1].review || "";
      const m = prevReview.match(/violations:\s*\[([^\]]*)\]/);
      if (m && m[1].trim()) {
        const viols = m[1].split(",").map(s=>s.replace(/['"]/g,"").trim()).filter(Boolean);
        focusPerAttempt.push(viols);
      } else {
        focusPerAttempt.push([]);
      }
    }
  });

  const frameDiv = document.createElement("div");
  frameDiv.className = "frame-block";

  const isClean = finalViolations.length === 0;
  const badge = isClean
    ? `<span style="color:#60f0a0;font-size:10px">✓ clean</span>`
    : `<span style="color:#f06060;font-size:10px">⚠ ${finalViolations.length} violations remain</span>`;

  frameDiv.innerHTML = `<div class="frame-label">
    Frame Id: ${frameId} &nbsp;${badge}
  </div>`;

  attempts.forEach((att, ai) => {
    const focus = focusPerAttempt[ai];
    const isRetry = att.review && att.review.includes("retry");
    const isDone  = att.review && att.review.includes("done");

    let reviewViolations = [];
    if (att.review) {
      const m = att.review.match(/violations:\s*\[([^\]]*)\]/);
      if (m && m[1].trim()) {
        reviewViolations = m[1].split(",").map(s=>s.replace(/['"]/g,"").trim()).filter(Boolean);
      }
    }

    const row = document.createElement("div");
    row.className = "attempt-row";

    let html = `<div class="attempt-num">Attempt ${ai+1}</div>`;
    html += `<div class="cards">`;

    // ── Vision Agent column: box + focus_areas below it ──
    const visionId = `vs-${frameId}-${ai}`;
    const focusText = `focus_areas: [${focus.join(", ")}]`;
    html += `<div class="node-col">
      <div class="node node-vision" id="${visionId}">
        <div class="node-title">Vision Agent</div>
      </div>
      <div class="below-text focus-text">${focusText}</div>
    </div>`;

    html += `<div class="arr-wrap"><div class="arr-label">Evaluated by</div><div class="arr">→</div></div>`;

    // ── Review column: box + violations below it (+ Done label if final) ──
    if (att.review) {
      let cls = isRetry ? "node-retry" : (isClean && isDone ? "node-done-clean" : "node-done-viol");
      let scoreHtml = "";

      if (isDone && Object.keys(finalScores).length > 0) {
        scoreHtml = `<div class="score-strip">`;
        CATS.forEach(cat => {
          const s = finalScores[cat];
          if (s === undefined) return;
          const scCls = s >= 1.0 ? "sc-ok" : s >= 0.5 ? "sc-mid" : "sc-bad";
          scoreHtml += `<span class="sc ${scCls}">${CAT_SHORT[cat]}:${s}</span>`;
        });
        scoreHtml += `</div>`;
      }

      const violText = `violations: [${reviewViolations.join(", ")}]`;
      const doneLine = isDone
        ? `<div class="below-text done-text">${isClean ? "✓ Done" : "Done"}</div>`
        : "";

      const reviewId = `rv-${frameId}-${ai}`;
      html += `<div class="node-col">
        <div class="node ${cls}" id="${reviewId}">
          <div class="node-title">Review</div>
          ${scoreHtml}
        </div>
        <div class="below-text viol-text">${violText}</div>
        ${doneLine}
      </div>`;
    }

    html += `</div>`; // cards

    row.innerHTML = html;
    frameDiv.appendChild(row);

    if (ai < attempts.length - 1) {
      const div = document.createElement("div");
      div.className = "divider";
      frameDiv.appendChild(div);
    }

    if (isRetry && ai + 1 < attempts.length) {
      retryConnectors.push({ from: `rv-${frameId}-${ai}`, to: `vs-${frameId}-${ai+1}` });
    }
  });

  timeline.appendChild(frameDiv);

  if (fi < frames.length - 1) {
    const sep = document.createElement("div");
    sep.style.cssText = "height:1px;background:#1e2130;margin:0 0 0 0";
    timeline.appendChild(sep);
  }
});

// ── Draw retry connector arrows (Review node -> next attempt's Vision node) ──
function drawRetryConnectors() {
  const svg = document.getElementById("connectorSvg");
  const docW = Math.max(document.body.scrollWidth, document.documentElement.scrollWidth);
  const docH = Math.max(document.body.scrollHeight, document.documentElement.scrollHeight);
  svg.setAttribute("width", docW);
  svg.setAttribute("height", docH);
  svg.setAttribute("viewBox", `0 0 ${docW} ${docH}`);

  svg.innerHTML = `
    <defs>
      <marker id="retryArrowHead" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
        <path d="M0,0 L6,3 L0,6 Z" fill="#c76b00" />
      </marker>
    </defs>
  `;

  function docRect(el) {
    const r = el.getBoundingClientRect();
    return {
      left: r.left + window.scrollX, right: r.right + window.scrollX,
      top: r.top + window.scrollY, bottom: r.bottom + window.scrollY,
      width: r.width, height: r.height
    };
  }

  retryConnectors.forEach(({ from, to }) => {
    const fromEl = document.getElementById(from);
    const toEl = document.getElementById(to);
    if (!fromEl || !toEl) return;

    const fr = docRect(fromEl);
    const tr = docRect(toEl);

    const startX = fr.right;
    const startY = fr.top + fr.height / 2;
    const endX = tr.left + tr.width / 2;
    const endY = tr.top;

    const cp1x = startX + 55, cp1y = startY;
    const cp2x = endX, cp2y = endY - 45;

    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("d", `M ${startX},${startY} C ${cp1x},${cp1y} ${cp2x},${cp2y} ${endX},${endY}`);
    path.setAttribute("stroke", "#c76b00");
    path.setAttribute("stroke-width", "1.8");
    path.setAttribute("stroke-dasharray", "5,4");
    path.setAttribute("fill", "none");
    path.setAttribute("marker-end", "url(#retryArrowHead)");
    svg.appendChild(path);

    const midX = (startX + cp1x + cp2x + endX) / 4;
    const midY = (startY + endY) / 2 - 8;

    const labelBg = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    labelBg.setAttribute("x", midX - 18);
    labelBg.setAttribute("y", midY - 9);
    labelBg.setAttribute("width", 36);
    labelBg.setAttribute("height", 14);
    labelBg.setAttribute("fill", "#ffffff");
    labelBg.setAttribute("stroke", "#c76b00");
    labelBg.setAttribute("stroke-width", "0.8");
    labelBg.setAttribute("rx", "3");
    svg.appendChild(labelBg);

    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("x", midX);
    text.setAttribute("y", midY + 3.5);
    text.setAttribute("text-anchor", "middle");
    text.setAttribute("font-size", "9.5");
    text.setAttribute("font-family", "Arial, sans-serif");
    text.setAttribute("font-weight", "bold");
    text.setAttribute("fill", "#c76b00");
    text.textContent = "Retry";
    svg.appendChild(text);
  });
}

drawRetryConnectors();
</script>
</body>
</html>""";

os.makedirs(os.path.dirname(OUTPUT_HTML), exist_ok=True)
with open(OUTPUT_HTML, "w") as f:
    f.write(html)
print(f"HTML saved → {OUTPUT_HTML}")

# ── PNG via playwright ─────────────────────────────────────────
try:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1300, "height": 900})
        page.goto(f"file://{OUTPUT_HTML}")
        page.wait_for_timeout(600)
        page.screenshot(path=OUTPUT_PNG, full_page=True)
        browser.close()
    print(f"PNG saved → {OUTPUT_PNG}")
except ImportError:
    print("playwright not installed — HTML only.")
    print("Run: pip install playwright && playwright install chromium")