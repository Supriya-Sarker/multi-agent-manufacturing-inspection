import json
import os

MEMORY_PATH = "/Users/supriya/Documents/data/agentic_system/multi-agent-manufacturing-inspection/training/reviewer_logs/run_memory.json"
OUTPUT_HTML = "/Users/supriya/Documents/data/agentic_system/multi-agent-manufacturing-inspection/training/results/memory_visual.html"
OUTPUT_PNG  = "/Users/supriya/Documents/data/agentic_system/multi-agent-manufacturing-inspection/training/results/memory_visual.png"

with open(MEMORY_PATH, "r") as f:
    raw = json.load(f)

# Filter out old-format keys — only keep real frame IDs (numeric strings)
all_memory = {k: v for k, v in raw.items() if k.isdigit()}

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
.frame-block { display: flex; flex-direction: column; gap: 0; }
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
  left: 60px;
  top: -20px;
  width: 2px;
  height: 20px;
  background: #c0c0c0;
}

/* ── Attempt index ── */
.attempt-num {
  width: 52px;
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

/* ── Arrow ── */
.arr {
  display: flex;
  align-items: center;
  padding: 0 6px;
  color: #3a3d4a;
  font-size: 18px;
  padding-top: 8px;
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

/* Focus injected */
.node-focus {
  background: #fffbe6;
  border-color: #111111;
  color: #5f5500;
  font-size: 10px;
  max-width: 220px;
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
  margin: 8px 0 0 52px;
}
</style>
</head>
<body>
<h1>WAAM Inspection — Decision Flow</h1>
<p class="sub">State · Focus Areas · Decisions per attempt · Generated from run_memory.json</p>

<div class="summary" id="summary"></div>
<div class="timeline" id="timeline"></div>

<script>
const CATS = ["Weld State","Arc Stability","Spatter Particle Size","Spatter Volume",
              "Melt Pool Length","Melt Pool Stability","Quality","Justification"];
const CAT_SHORT = {"Weld State":"WS","Arc Stability":"ARC","Spatter Particle Size":"SPS",
  "Spatter Volume":"SV","Melt Pool Length":"MPL","Melt Pool Stability":"MPS",
  "Quality":"Q","Justification":"JUST"};

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

frames.forEach(([frameId, frame], fi) => {
  const route = frame.route || [];
  const finalViolations = frame.violations || [];
  const finalScores = frame.category_scores || {};
  const retries = frame.retry_count || 0;

  // Parse route into attempts
  // Each attempt = vision_node + review_node pair
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

  // Build focus areas per attempt from final state
  // Attempt 1 has no focus (fresh start)
  // Attempts 2+ had focus injected from previous review
  // We reconstruct focus from the route violations text
  const focusPerAttempt = [];
  attempts.forEach((att, i) => {
    if (i === 0) {
      focusPerAttempt.push([]);
    } else {
      // Extract violations from previous review step
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
    Frame ${fi+1} &nbsp;${badge}
    <span class="fid">${frameId}</span>
  </div>`;

  attempts.forEach((att, ai) => {
    const focus = focusPerAttempt[ai];
    const isRetry = att.review && att.review.includes("retry");
    const isDone  = att.review && att.review.includes("done");

    // Extract violations from this review
    let reviewViolations = [];
    if (att.review) {
      const m = att.review.match(/violations:\s*\[([^\]]*)\]/);
      if (m && m[1].trim()) {
        reviewViolations = m[1].split(",").map(s=>s.replace(/['"]/g,"").trim()).filter(Boolean);
      }
    }

    const row = document.createElement("div");
    row.className = "attempt-row";

    // Attempt number
    let html = `<div class="attempt-num">A${ai+1}</div>`;
    html += `<div class="cards">`;

    // Focus block (only if there are focus areas)
    if (focus.length > 0) {
      html += `<div class="node node-focus">
        <div class="node-title">Focus Injected</div>
        <div class="node-body">${focus.map(f=>`→ ${f}`).join("<br>")}</div>
      </div>`;
      html += `<div class="arr">→</div>`;
    }

    // Vision node
    html += `<div class="node node-vision">
      <div class="node-title">Vision Agent</div>
      <div class="node-body">Attempt #${ai+1}</div>
    </div>`;
    html += `<div class="arr">→</div>`;

    // Review node
    if (att.review) {
      let cls = isRetry ? "node-retry" : (isClean && isDone ? "node-done-clean" : "node-done-viol");
      let label = isRetry ? "↻ Review → Retry" : "✓ Review → Done";
      let scoreHtml = "";

      // On final attempt show scores
      if (isDone && Object.keys(finalScores).length > 0) {
        scoreHtml = `<div class="score-strip">`;
        CATS.forEach(cat => {
          const s = finalScores[cat];
          if (s === undefined) return;
          const scCls = s >= 1.0 ? "sc-ok" : s >= 0.5 ? "sc-mid" : "sc-bad";
          scoreHtml += `<span class="sc ${scCls}">${CAT_SHORT[cat]}:${s}</span>`;
        });
        scoreHtml += `</div>`;
      } else if (reviewViolations.length > 0) {
        scoreHtml = `<div style="margin-top:4px;font-size:10px;color:#f0a060">${reviewViolations.join(", ")}</div>`;
      }

      html += `<div class="node ${cls}">
        <div class="node-title">${label}</div>
        <div class="node-body">
          ${reviewViolations.length > 0 && !isDone ? reviewViolations.length + " violations" : ""}
          ${scoreHtml}
        </div>
      </div>`;
    }

    html += `</div>`; // cards

    row.innerHTML = html;
    frameDiv.appendChild(row);

    // Divider between attempts
    if (ai < attempts.length - 1) {
      const div = document.createElement("div");
      div.className = "divider";
      frameDiv.appendChild(div);
    }
  });

  timeline.appendChild(frameDiv);

  // Gap between frames
  if (fi < frames.length - 1) {
    const sep = document.createElement("div");
    sep.style.cssText = "height:1px;background:#1e2130;margin:0 0 0 0";
    timeline.appendChild(sep);
  }
});
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