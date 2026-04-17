"""
Project A — FastAPI Web Application
Acmera Knowledge Assistant (RAG Pipeline)

Endpoints:
  GET  /             → HTML UI
  GET  /health       → ALB health check
  POST /ask          → run RAG pipeline, return answer + metadata
  GET  /cache/stats  → semantic cache statistics
  DELETE /cache      → clear semantic cache

Run locally:
    uvicorn app:app --reload --port 8000
"""
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Acmera RAG — Project A", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Semantic cache: Redis in production, in-memory locally ──────────────────
from scripts.semantic_cache import SemanticCache, get_cache

_redis_url = os.getenv("REDIS_URL")
if _redis_url:
    import redis as _redis_lib
    _r = _redis_lib.from_url(_redis_url, decode_responses=True)
    _cache = SemanticCache(threshold=0.92, ttl_hours=24, redis_client=_r)
else:
    _cache = get_cache()


# ── Request / Response models ───────────────────────────────────────────────
class AskRequest(BaseModel):
    query: str
    mode: str = "advanced"
    use_cache: bool = True


class FeedbackRequest(BaseModel):
    trace_id: str
    rating: int          # +1 = thumbs up, -1 = thumbs down
    query: str = ""
    comment: str = ""


# ── Endpoints ───────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "service": "project-a-rag"}


@app.get("/cache/stats")
def cache_stats():
    return _cache.stats()


@app.delete("/cache")
def clear_cache():
    _cache.clear()
    return {"message": "Cache cleared"}


@app.post("/ask")
def ask_endpoint(req: AskRequest):
    from scripts.rag import ask
    try:
        result = ask(req.query, mode=req.mode, use_cache=req.use_cache, cache=_cache)
        return {
            "query":           result["query"],
            "answer":          result["answer"],
            "mode":            result["mode"],
            "cache_hit":       result.get("cache_hit", False),
            "model_used":      result.get("model_used", "—"),
            "elapsed_seconds": result["elapsed_seconds"],
            "trace_id":        result.get("trace_id"),
            "chunks": [
                {
                    "doc_name":    c["doc_name"],
                    "chunk_index": c["chunk_index"],
                    "score":       round(
                        c.get("rerank_score", c.get("rrf_score", c.get("similarity", 0))), 4
                    ),
                }
                for c in result.get("retrieved_chunks", [])
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
def feedback_endpoint(req: FeedbackRequest):
    import psycopg2
    from langfuse import Langfuse

    # 1. LangFuse — attaches score to the trace for dashboard visibility
    try:
        lf = Langfuse()
        lf.score(
            trace_id=req.trace_id,
            name="user_feedback",
            value=req.rating,
            comment=req.comment or None,
        )
        lf.flush()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LangFuse error: {e}")

    # 2. RDS — persists feedback for SQL querying and future analysis
    try:
        conn = psycopg2.connect(
            host=os.getenv("PG_HOST", "localhost"),
            port=int(os.getenv("PG_PORT", "5433")),
            user=os.getenv("PG_USER", "workshop"),
            password=os.getenv("PG_PASSWORD", "workshop123"),
            dbname=os.getenv("PG_DATABASE", "acmera_kb"),
        )
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO feedback (trace_id, query, rating, comment, source)
               VALUES (%s, %s, %s, %s, 'project-a')""",
            (req.trace_id, req.query or None, req.rating, req.comment or None),
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def ui():
    return PROJECT_A_HTML


# ── HTML UI ─────────────────────────────────────────────────────────────────
PROJECT_A_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Acmera Knowledge Assistant</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #0f172a;
      color: #e2e8f0;
      min-height: 100vh;
    }

    /* ── Header ── */
    header {
      background: #1e293b;
      border-bottom: 1px solid #334155;
      padding: 16px 32px;
      display: flex;
      align-items: center;
      gap: 12px;
    }
    header .logo { font-size: 20px; font-weight: 700; color: #6366f1; }
    header .subtitle { font-size: 13px; color: #64748b; margin-left: 4px; }
    header .badge {
      margin-left: auto;
      font-size: 11px;
      padding: 3px 10px;
      border-radius: 99px;
      background: #1e293b;
      border: 1px solid #334155;
      color: #94a3b8;
    }

    /* ── Layout ── */
    .container { max-width: 1100px; margin: 0 auto; padding: 32px 24px; }
    .layout { display: grid; grid-template-columns: 380px 1fr; gap: 24px; }
    @media (max-width: 768px) { .layout { grid-template-columns: 1fr; } }

    /* ── Cards ── */
    .card {
      background: #1e293b;
      border: 1px solid #334155;
      border-radius: 12px;
      padding: 20px;
    }
    .card-title {
      font-size: 12px;
      font-weight: 600;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: #64748b;
      margin-bottom: 16px;
    }

    /* ── Controls ── */
    textarea {
      width: 100%;
      background: #0f172a;
      border: 1px solid #334155;
      border-radius: 8px;
      color: #e2e8f0;
      font-size: 14px;
      padding: 12px;
      resize: vertical;
      min-height: 100px;
      outline: none;
      transition: border-color 0.2s;
    }
    textarea:focus { border-color: #6366f1; }

    label.section-label {
      display: block;
      font-size: 12px;
      color: #64748b;
      margin: 16px 0 8px;
    }

    .mode-group { display: flex; gap: 6px; flex-wrap: wrap; }
    .mode-btn {
      padding: 6px 14px;
      border-radius: 6px;
      border: 1px solid #334155;
      background: #0f172a;
      color: #94a3b8;
      font-size: 13px;
      cursor: pointer;
      transition: all 0.15s;
    }
    .mode-btn:hover { border-color: #6366f1; color: #e2e8f0; }
    .mode-btn.active { background: #6366f1; border-color: #6366f1; color: white; }

    .toggle-row {
      display: flex;
      align-items: center;
      gap: 10px;
      margin-top: 16px;
    }
    .toggle {
      position: relative;
      width: 40px;
      height: 22px;
      cursor: pointer;
    }
    .toggle input { opacity: 0; width: 0; height: 0; }
    .toggle-slider {
      position: absolute;
      inset: 0;
      background: #334155;
      border-radius: 22px;
      transition: 0.2s;
    }
    .toggle-slider::before {
      content: "";
      position: absolute;
      width: 16px; height: 16px;
      left: 3px; bottom: 3px;
      background: white;
      border-radius: 50%;
      transition: 0.2s;
    }
    input:checked + .toggle-slider { background: #6366f1; }
    input:checked + .toggle-slider::before { transform: translateX(18px); }
    .toggle-label { font-size: 13px; color: #94a3b8; }

    .ask-btn {
      width: 100%;
      margin-top: 20px;
      padding: 12px;
      background: #6366f1;
      color: white;
      border: none;
      border-radius: 8px;
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.2s;
    }
    .ask-btn:hover { background: #4f46e5; }
    .ask-btn:disabled { background: #334155; color: #64748b; cursor: not-allowed; }

    /* ── Answer area ── */
    .answer-placeholder {
      color: #475569;
      font-size: 14px;
      text-align: center;
      padding: 60px 0;
    }

    .meta-strip {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 16px;
    }
    .meta-pill {
      font-size: 11px;
      padding: 3px 10px;
      border-radius: 99px;
      border: 1px solid #334155;
      color: #94a3b8;
    }
    .meta-pill.hit    { border-color: #22c55e; color: #22c55e; }
    .meta-pill.miss   { border-color: #334155; color: #64748b; }
    .meta-pill.mini   { border-color: #22c55e; color: #22c55e; }
    .meta-pill.full   { border-color: #f59e0b; color: #f59e0b; }
    .meta-pill.cached { border-color: #6366f1; color: #6366f1; }

    .answer-text {
      font-size: 15px;
      line-height: 1.7;
      color: #e2e8f0;
      white-space: pre-wrap;
    }

    .sources-toggle {
      margin-top: 20px;
      font-size: 12px;
      color: #6366f1;
      cursor: pointer;
      background: none;
      border: none;
      padding: 0;
    }
    .sources-toggle:hover { text-decoration: underline; }

    .sources-list { margin-top: 10px; display: none; }
    .sources-list.open { display: block; }
    .source-item {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 6px 10px;
      margin-bottom: 4px;
      background: #0f172a;
      border-radius: 6px;
      font-size: 12px;
    }
    .source-doc { color: #94a3b8; }
    .source-score { color: #6366f1; font-weight: 600; }

    /* ── Spinner ── */
    .spinner {
      display: inline-block;
      width: 18px; height: 18px;
      border: 2px solid #334155;
      border-top-color: #6366f1;
      border-radius: 50%;
      animation: spin 0.7s linear infinite;
      vertical-align: middle;
      margin-right: 8px;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    .loading-text { color: #64748b; font-size: 14px; padding: 40px 0; text-align: center; }

    /* ── Error ── */
    .error-box {
      background: #450a0a;
      border: 1px solid #7f1d1d;
      border-radius: 8px;
      padding: 12px 16px;
      font-size: 14px;
      color: #fca5a5;
    }

    /* ── Feedback ── */
    .feedback-row {
      display: flex;
      align-items: center;
      gap: 10px;
      margin-top: 20px;
      padding-top: 16px;
      border-top: 1px solid #334155;
    }
    .feedback-label { font-size: 12px; color: #64748b; }
    .thumb-btn {
      background: none;
      border: 1px solid #334155;
      border-radius: 8px;
      color: #94a3b8;
      font-size: 18px;
      width: 38px; height: 38px;
      cursor: pointer;
      transition: all 0.15s;
      display: flex; align-items: center; justify-content: center;
    }
    .thumb-btn:hover { border-color: #6366f1; color: #e2e8f0; }
    .thumb-btn.selected-up   { background: #14532d; border-color: #22c55e; color: #22c55e; }
    .thumb-btn.selected-down { background: #450a0a; border-color: #ef4444; color: #ef4444; }
    .feedback-thanks { font-size: 12px; color: #22c55e; display: none; }
  </style>
</head>
<body>
  <header>
    <span class="logo">Acmera</span>
    <span class="subtitle">Knowledge Assistant</span>
    <span class="badge">Project A — RAG Pipeline</span>
  </header>

  <div class="container">
    <div class="layout">

      <!-- Controls panel -->
      <div class="card">
        <div class="card-title">Query</div>
        <textarea id="queryInput" placeholder="Ask anything about Acmera policies, orders, membership...&#10;&#10;e.g. What is the return window for Premium Gold members during Diwali sale?"></textarea>

        <label class="section-label">Retrieval Mode</label>
        <div class="mode-group" id="modeGroup">
          <button class="mode-btn" data-mode="dense">Dense</button>
          <button class="mode-btn" data-mode="hybrid">Hybrid</button>
          <button class="mode-btn active" data-mode="advanced">Advanced</button>
        </div>
        <p style="font-size:11px;color:#475569;margin-top:8px;">
          Dense → Hybrid (BM25+RRF) → Advanced (+ Cohere rerank)
        </p>

        <div class="toggle-row">
          <label class="toggle">
            <input type="checkbox" id="cacheToggle" checked />
            <span class="toggle-slider"></span>
          </label>
          <span class="toggle-label">Semantic cache</span>
        </div>

        <button class="ask-btn" id="askBtn" onclick="runQuery()">Ask</button>

        <div style="margin-top:20px;padding-top:16px;border-top:1px solid #334155;">
          <div class="card-title">Cache</div>
          <div id="cacheStats" style="font-size:12px;color:#64748b;">—</div>
          <button onclick="loadCacheStats()"
            style="margin-top:8px;font-size:12px;color:#6366f1;background:none;border:none;cursor:pointer;padding:0;">
            Refresh stats
          </button>
          &nbsp;·&nbsp;
          <button onclick="clearCache()"
            style="font-size:12px;color:#ef4444;background:none;border:none;cursor:pointer;padding:0;">
            Clear cache
          </button>
        </div>
      </div>

      <!-- Answer panel -->
      <div class="card">
        <div class="card-title">Answer</div>
        <div id="answerArea">
          <div class="answer-placeholder">Ask a question to see the answer here.</div>
        </div>
      </div>

    </div>
  </div>

  <script>
    let selectedMode = "advanced";
    let sourcesOpen = false;
    let currentTraceId = null;

    // Mode buttons
    document.querySelectorAll(".mode-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".mode-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        selectedMode = btn.dataset.mode;
      });
    });

    async function runQuery() {
      const query = document.getElementById("queryInput").value.trim();
      if (!query) return;

      const useCache = document.getElementById("cacheToggle").checked;
      const btn = document.getElementById("askBtn");
      btn.disabled = true;
      btn.textContent = "Thinking...";

      document.getElementById("answerArea").innerHTML =
        '<div class="loading-text"><span class="spinner"></span>Running pipeline...</div>';

      try {
        const res = await fetch("/ask", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query, mode: selectedMode, use_cache: useCache }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Request failed");
        renderAnswer(data);
        loadCacheStats();
      } catch (err) {
        document.getElementById("answerArea").innerHTML =
          `<div class="error-box">${err.message}</div>`;
      } finally {
        btn.disabled = false;
        btn.textContent = "Ask";
      }
    }

    function renderAnswer(data) {
      currentTraceId = data.trace_id || null;
      const cacheClass = data.cache_hit ? "hit" : "miss";
      const cacheLabel = data.cache_hit ? "Cache HIT" : "Cache miss";
      let modelClass = "meta-pill";
      let modelLabel = data.model_used;
      if (data.model_used === "cached") {
        modelClass += " cached"; modelLabel = "Served from cache";
      } else if (data.model_used && data.model_used.includes("mini")) {
        modelClass += " mini";
      } else if (data.model_used && data.model_used.includes("4o")) {
        modelClass += " full";
      }

      const chunksHtml = data.chunks.length > 0 ? `
        <button class="sources-toggle" onclick="toggleSources()">
          ▸ ${data.chunks.length} source${data.chunks.length > 1 ? "s" : ""} used
        </button>
        <div class="sources-list" id="sourcesList">
          ${data.chunks.map(c => `
            <div class="source-item">
              <span class="source-doc">${c.doc_name} · chunk ${c.chunk_index}</span>
              <span class="source-score">${c.score}</span>
            </div>`).join("")}
        </div>` : "";

      const feedbackHtml = currentTraceId ? `
        <div class="feedback-row">
          <span class="feedback-label">Was this helpful?</span>
          <button class="thumb-btn" id="thumbUp"   onclick="sendFeedback(1)"  title="Helpful">👍</button>
          <button class="thumb-btn" id="thumbDown" onclick="sendFeedback(-1)" title="Not helpful">👎</button>
          <span class="feedback-thanks" id="feedbackThanks">Thanks for your feedback!</span>
        </div>` : "";

      document.getElementById("answerArea").innerHTML = `
        <div class="meta-strip">
          <span class="meta-pill ${cacheClass}">${cacheLabel}</span>
          <span class="${modelClass}">${modelLabel}</span>
          <span class="meta-pill">${data.mode} mode</span>
          <span class="meta-pill">${data.elapsed_seconds}s</span>
        </div>
        <div class="answer-text">${escapeHtml(data.answer)}</div>
        ${chunksHtml}
        ${feedbackHtml}
      `;
    }

    function toggleSources() {
      sourcesOpen = !sourcesOpen;
      const list = document.getElementById("sourcesList");
      const btn = document.querySelector(".sources-toggle");
      if (list) {
        list.classList.toggle("open", sourcesOpen);
        if (btn) btn.textContent = (sourcesOpen ? "▾ " : "▸ ") + btn.textContent.slice(2);
      }
    }

    async function loadCacheStats() {
      try {
        const res = await fetch("/cache/stats");
        const s = await res.json();
        document.getElementById("cacheStats").innerHTML =
          `Backend: <b>${s.backend}</b> &nbsp;|&nbsp; ` +
          `Size: <b>${s.cache_size}</b> &nbsp;|&nbsp; ` +
          `Hit rate: <b>${(s.hit_rate * 100).toFixed(0)}%</b> ` +
          `(${s.cache_hits}/${s.total_queries})`;
      } catch { }
    }

    async function clearCache() {
      await fetch("/cache", { method: "DELETE" });
      loadCacheStats();
    }

    async function sendFeedback(rating) {
      if (!currentTraceId) return;
      const upBtn   = document.getElementById("thumbUp");
      const downBtn = document.getElementById("thumbDown");
      const thanks  = document.getElementById("feedbackThanks");
      if (!upBtn) return;

      upBtn.disabled = true; downBtn.disabled = true;
      if (rating === 1)  upBtn.classList.add("selected-up");
      else               downBtn.classList.add("selected-down");

      try {
        const query = document.getElementById("queryInput").value.trim();
        await fetch("/feedback", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ trace_id: currentTraceId, rating, query }),
        });
        if (thanks) thanks.style.display = "inline";
      } catch (e) {
        console.error("Feedback error:", e);
      }
    }

    function escapeHtml(str) {
      return str.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
    }

    // Enter key submits
    document.getElementById("queryInput").addEventListener("keydown", e => {
      if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) runQuery();
    });

    // Load stats on page load
    loadCacheStats();
  </script>
</body>
</html>"""
