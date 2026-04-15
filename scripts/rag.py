"""
RAG pipeline — Instructor version, Session 7 complete.

Full pipeline progression (all modes supported):
  dense    — Session 1 baseline (pure vector similarity)
  hybrid   — Session 3: BM25 + dense + RRF fusion
  advanced — Session 4: hybrid + Cohere rerank + context engineering
  cached   — Session 5: semantic cache check before advanced pipeline

Session 5 additions:
  - model_router: routes each query to gpt-4o-mini (simple) or gpt-4o (complex)
  - semantic_cache: skips retrieval + generation on cache hit
  - result dict now includes cache_hit and model_used fields

Session 7 additions:
  - LiteLLM replaces direct OpenAI calls in generate() — provider-agnostic
    routing (OpenAI, Anthropic, Google, any provider via one interface)
  - guardrails=True param on ask(): runs check_input before pipeline,
    check_output on answer — blocked queries never hit retrieval
  - ask_structured(): returns a typed RAGResponse via Instructor instead
    of a raw dict — confidence level, sources list, escalation flag

Run:
    python -m scripts.rag                              # dense
    python -m scripts.rag --mode hybrid               # Session 3
    python -m scripts.rag --mode advanced             # Session 4
    python -m scripts.rag --mode advanced --cache     # Session 5: with cache
    python -m scripts.rag --mode advanced --guardrails # Session 7: with guardrails
"""
import os
import json
import time
import argparse
import litellm
import instructor
from openai import OpenAI
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
import psycopg2
from pgvector.psycopg2 import register_vector
from rank_bm25 import BM25Okapi
from pydantic import BaseModel
from typing import Literal
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich import box

from scripts.model_router import route_model
from scripts.semantic_cache import get_cache
from scripts.guardrails import check_input, check_output

load_dotenv()

client = OpenAI()
instructor_client = instructor.from_openai(OpenAI())
langfuse = Langfuse()
console = Console()


# =============================================================================
# SESSION 7: STRUCTURED RESPONSE MODEL
# ask_structured() returns this instead of a raw dict.
# Instructor extracts it from the generated answer in one LLM call.
# =============================================================================

class RAGResponse(BaseModel):
    answer: str
    confidence: Literal["high", "medium", "low"]
    sources: list[str]
    safe: bool
    requires_escalation: bool

TOP_K = 5
BM25_CANDIDATES = TOP_K * 3
GENERATION_MODEL = "gpt-4o-mini"  # fallback; route_model() overrides per query

SYSTEM_PROMPT = """You are a helpful customer support assistant for Acmera, an Indian e-commerce company.
Answer the customer's question based on the provided context from our documentation.

Rules:
- Only answer based on the provided context. If the context doesn't contain enough information, say so.
- Be specific and cite relevant policy details (days, amounts, conditions).
- If the question involves membership tiers, check the context for tier-specific policies.
- Be concise but thorough.

Context from Acmera documentation:
{context}"""


def get_connection():
    conn = psycopg2.connect(
        host=os.getenv("PG_HOST", "localhost"),
        port=os.getenv("PG_PORT", "5433"),
        user=os.getenv("PG_USER", "workshop"),
        password=os.getenv("PG_PASSWORD", "workshop123"),
        dbname=os.getenv("PG_DATABASE", "acmera_kb"),
    )
    register_vector(conn)
    return conn


@observe(name="query_embedding")
def embed_query(query):
    response = client.embeddings.create(model="text-embedding-3-small", input=query)
    return response.data[0].embedding


@observe(name="retrieval_dense")
def retrieve(query_embedding, top_k=TOP_K):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """SELECT id, doc_name, chunk_index, content, metadata,
                  1 - (embedding <=> %s::vector) AS similarity
           FROM chunks ORDER BY embedding <=> %s::vector LIMIT %s""",
        (query_embedding, query_embedding, top_k),
    )
    results = [{
        "id": row[0], "doc_name": row[1], "chunk_index": row[2],
        "content": row[3],
        "metadata": row[4] if isinstance(row[4], dict) else json.loads(row[4]),
        "similarity": round(float(row[5]), 4),
    } for row in cur.fetchall()]
    cur.close()
    conn.close()
    langfuse_context.update_current_observation(metadata={
        "mode": "dense", "top_k": top_k,
        "results": [{"doc_name": r["doc_name"], "similarity": r["similarity"]} for r in results],
    })
    return results


def _load_all_chunks():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, doc_name, chunk_index, content, metadata FROM chunks ORDER BY id")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [{
        "id": row[0], "doc_name": row[1], "chunk_index": row[2], "content": row[3],
        "metadata": row[4] if isinstance(row[4], dict) else json.loads(row[4]),
    } for row in rows]


def build_bm25_index():
    all_chunks = _load_all_chunks()
    return BM25Okapi([c["content"].lower().split() for c in all_chunks]), all_chunks


def bm25_retrieve(query, bm25, all_chunks, top_k=BM25_CANDIDATES):
    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    results = []
    for idx, score in ranked:
        if score > 0:
            chunk = all_chunks[idx].copy()
            chunk["bm25_score"] = round(float(score), 4)
            chunk["similarity"] = 0.0
            results.append(chunk)
    return results


def reciprocal_rank_fusion(dense_results, bm25_results, top_k=TOP_K, k=60):
    scores, chunk_map = {}, {}
    for rank, chunk in enumerate(dense_results):
        cid = chunk["id"]
        scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank + 1)
        chunk_map[cid] = chunk
    for rank, chunk in enumerate(bm25_results):
        cid = chunk["id"]
        scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank + 1)
        if cid not in chunk_map:
            chunk_map[cid] = chunk
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [{**chunk_map[cid], "rrf_score": round(rrf, 6)} for cid, rrf in ranked]


@observe(name="retrieval_hybrid")
def hybrid_retrieve(query, query_embedding, top_k=TOP_K):
    bm25, all_chunks = build_bm25_index()
    dense = retrieve.__wrapped__(query_embedding, top_k=BM25_CANDIDATES)
    bm25_results = bm25_retrieve(query, bm25, all_chunks, top_k=BM25_CANDIDATES)
    fused = reciprocal_rank_fusion(dense, bm25_results, top_k=top_k)
    langfuse_context.update_current_observation(metadata={
        "mode": "hybrid", "fused": len(fused),
        "results": [{"doc_name": r["doc_name"], "rrf_score": r.get("rrf_score")} for r in fused],
    })
    return fused


@observe(name="context_assembly")
def assemble_context(retrieved_chunks):
    parts = [f"[Source: {c['doc_name']}, Chunk {c['chunk_index']}]\n{c['content']}" for c in retrieved_chunks]
    context = "\n\n---\n\n".join(parts)
    langfuse_context.update_current_observation(metadata={
        "num_chunks": len(retrieved_chunks), "total_chars": len(context),
    })
    return context


@observe(name="generation")
def generate(query, context, model: str = None):
    """
    Generate an answer using LiteLLM (Session 7).

    LiteLLM is a provider-agnostic wrapper — the same call works for
    OpenAI, Anthropic Claude, Google Gemini, Mistral, and local models.
    model_router still picks the model; LiteLLM executes it regardless
    of which provider that model belongs to.

    Examples of what model strings LiteLLM accepts:
      "gpt-4o-mini"                   → OpenAI
      "claude-3-haiku-20240307"       → Anthropic
      "gemini/gemini-1.5-flash"       → Google
    """
    if model is None:
        model = route_model(query)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
        {"role": "user", "content": query},
    ]
    # litellm.completion() is a drop-in for openai.chat.completions.create()
    response = litellm.completion(
        model=model, messages=messages, temperature=0, max_tokens=1000,
    )
    answer = response.choices[0].message.content
    langfuse_context.update_current_observation(
        input=messages, output=answer,
        metadata={"model": model,
                  "prompt_tokens": response.usage.prompt_tokens,
                  "completion_tokens": response.usage.completion_tokens},
        usage={"input": response.usage.prompt_tokens, "output": response.usage.completion_tokens,
               "total": response.usage.total_tokens, "unit": "TOKENS"},
    )
    return answer, model


@observe(name="rag_pipeline")
def ask(query, mode="dense", use_cache: bool = False, cache=None,
        guardrails: bool = False):
    """
    Run the RAG pipeline.

    mode:       "dense" | "hybrid" | "advanced"
    use_cache:  if True, check semantic cache before retrieval.
                On a hit, skips embedding → retrieval → generation entirely.
                On a miss, runs the pipeline and stores the result.
    cache:      optional SemanticCache instance. Falls back to in-memory singleton.
    guardrails: if True (Session 7), run check_input before the pipeline and
                check_output on the answer. Blocked queries return immediately
                without touching retrieval or generation.
    """
    start_time = time.time()
    langfuse_context.update_current_trace(
        input=query,
        metadata={"pipeline": f"rag_{mode}", "cache_enabled": use_cache,
                  "guardrails": guardrails},
    )

    # ── Session 7: input guardrail ────────────────────────────────────────────
    if guardrails:
        in_guard = check_input(query)
        if not in_guard.safe:
            elapsed = round(time.time() - start_time, 2)
            langfuse_context.update_current_trace(
                output="[BLOCKED]",
                metadata={"blocked": True, "reason": in_guard.rejection_reason,
                          "elapsed_seconds": elapsed},
            )
            langfuse.flush()
            return {
                "query": query, "answer": in_guard.rejection_reason,
                "retrieved_chunks": [], "context": "",
                "trace_id": langfuse_context.get_current_trace_id(),
                "elapsed_seconds": elapsed, "mode": mode,
                "cache_hit": False, "model_used": None,
                "blocked": True,
            }
        # Use anonymised query downstream so PII doesn't enter retrieval/logs
        query = in_guard.anonymized_query
    # ─────────────────────────────────────────────────────────────────────────

    query_embedding = embed_query(query)

    # ── Session 5: semantic cache check ──────────────────────────────────────
    # cache param: pass a SemanticCache instance from the caller (e.g. app.py
    # passes a Redis-backed instance). Falls back to in-memory singleton.
    if use_cache:
        _cache = cache if cache is not None else get_cache()
        cache_hit, cached_answer = _cache.check(query_embedding)
        if cache_hit:
            elapsed = round(time.time() - start_time, 2)
            langfuse_context.update_current_trace(
                output=cached_answer,
                metadata={"elapsed_seconds": elapsed, "cache_hit": True},
            )
            langfuse.flush()
            return {
                "query": query, "answer": cached_answer,
                "retrieved_chunks": [], "context": "",
                "trace_id": langfuse_context.get_current_trace_id(),
                "elapsed_seconds": elapsed, "mode": mode,
                "cache_hit": True, "model_used": "cached",
            }
    # ─────────────────────────────────────────────────────────────────────────

    if mode == "advanced":
        result = _ask_advanced(query, query_embedding, start_time)
    else:
        if mode == "hybrid":
            retrieved_chunks = hybrid_retrieve(query, query_embedding)
        else:
            retrieved_chunks = retrieve(query_embedding)

        context = assemble_context(retrieved_chunks)
        answer, model_used = generate(query, context)

        elapsed = round(time.time() - start_time, 2)
        langfuse_context.update_current_trace(
            output=answer,
            metadata={"elapsed_seconds": elapsed, "cache_hit": False, "model_used": model_used},
        )
        trace_id = langfuse_context.get_current_trace_id()
        langfuse.flush()

        result = {
            "query": query, "answer": answer, "retrieved_chunks": retrieved_chunks,
            "context": context, "trace_id": trace_id, "elapsed_seconds": elapsed, "mode": mode,
            "cache_hit": False, "model_used": model_used,
        }

    # Store in cache after pipeline runs (on miss)
    if use_cache:
        _cache = cache if cache is not None else get_cache()
        _cache.store(query_embedding, query, result["answer"])

    # ── Session 7: output guardrail ───────────────────────────────────────────
    if guardrails:
        out_guard = check_output(result["answer"])
        result["answer"] = out_guard.clean_answer
        result["output_safe"] = out_guard.safe
        result["pii_leaked"] = out_guard.pii_leaked
    # ─────────────────────────────────────────────────────────────────────────

    return result


def _ask_advanced(query, query_embedding, start_time):
    from scripts.reranker import rerank
    from scripts.context_assembler import assemble_advanced

    candidates = hybrid_retrieve(query, query_embedding, top_k=TOP_K * 2)
    reranked = rerank(query, candidates, top_n=TOP_K + 2)
    context, final_chunks = assemble_advanced(reranked, conn_fn=get_connection, max_chars=4000, expand_window=1)
    answer, model_used = generate(query, context)

    elapsed = round(time.time() - start_time, 2)
    langfuse_context.update_current_trace(
        output=answer, metadata={
            "elapsed_seconds": elapsed, "mode": "advanced",
            "cache_hit": False, "model_used": model_used,
            "stages": {"candidates": len(candidates), "reranked": len(reranked), "final": len(final_chunks)},
        }
    )
    trace_id = langfuse_context.get_current_trace_id()
    langfuse.flush()

    return {
        "query": query, "answer": answer, "retrieved_chunks": final_chunks,
        "context": context, "trace_id": trace_id, "elapsed_seconds": elapsed, "mode": "advanced",
        "cache_hit": False, "model_used": model_used,
        "pipeline_stages": {"candidates": len(candidates), "reranked": len(reranked), "final": len(final_chunks)},
    }


def ask_hybrid(query):
    return ask(query, mode="hybrid")


def ask_advanced(query):
    return ask(query, mode="advanced")


def ask_cached(query):
    """Run advanced pipeline with semantic cache. Session 5 showcase."""
    return ask(query, mode="advanced", use_cache=True)


def ask_structured(query: str) -> RAGResponse:
    """
    Session 7: Run the advanced pipeline and return a typed RAGResponse.

    Uses Instructor to extract a structured response from the generated answer
    in a single LLM call — no extra API call compared to ask().

    Returns:
        RAGResponse with answer, confidence, sources, safe, requires_escalation
    """
    result = ask(query, mode="advanced", guardrails=True)

    if result.get("blocked"):
        return RAGResponse(
            answer="This request could not be processed.",
            confidence="high",
            sources=[],
            safe=False,
            requires_escalation=False,
        )

    sources = list({c["doc_name"] for c in result.get("retrieved_chunks", [])})

    return instructor_client.chat.completions.create(
        model=route_model(query),
        response_model=RAGResponse,
        messages=[
            {
                "role": "system",
                "content": """Extract a structured RAGResponse from this customer support answer.

confidence:
  high   = answer is specific, complete, and directly references policy details
  medium = answer is mostly complete but has some uncertainty or gaps
  low    = answer is vague, context was insufficient, or the question was ambiguous

sources: extract document names that supported the answer from the sources list provided.

requires_escalation: True only if the answer explicitly cannot be resolved
  by policy alone and a human agent must intervene (billing disputes, security issues, etc.)
  False for standard policy questions, even complex ones.""",
            },
            {
                "role": "user",
                "content": (
                    f"Question: {query}\n\n"
                    f"Answer: {result['answer']}\n\n"
                    f"Sources used: {sources}"
                ),
            },
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dense", "hybrid", "advanced"], default="dense")
    parser.add_argument("--query", default="What is the return window for premium members during Diwali sale?")
    parser.add_argument("--cache", action="store_true", help="Enable semantic cache (Session 5)")
    parser.add_argument("--guardrails", action="store_true", help="Enable guardrails (Session 7)")
    parser.add_argument("--structured", action="store_true", help="Return RAGResponse via Instructor (Session 7)")
    args = parser.parse_args()

    if args.structured:
        response = ask_structured(args.query)
        console.print(f"\n[bold]Structured RAGResponse:[/]")
        console.print(f"  answer:               {response.answer[:80]}...")
        console.print(f"  confidence:           {response.confidence}")
        console.print(f"  sources:              {response.sources}")
        console.print(f"  safe:                 {response.safe}")
        console.print(f"  requires_escalation:  {response.requires_escalation}")
        import sys; sys.exit(0)

    result = ask(args.query, mode=args.mode, use_cache=args.cache, guardrails=args.guardrails)

    table = Table(title=f"RAG Result — mode: {result['mode']}", box=box.ROUNDED)
    table.add_column("Field", style="bold", width=14)
    table.add_column("Value")
    table.add_row("Mode", result["mode"])
    table.add_row("Model", result.get("model_used", "—"))
    table.add_row("Cache Hit", "[green]YES[/]" if result.get("cache_hit") else "no")
    table.add_row("Time", f"{result['elapsed_seconds']}s")
    if "pipeline_stages" in result:
        s = result["pipeline_stages"]
        table.add_row("Pipeline", f"{s['candidates']} candidates → {s['reranked']} reranked → {s['final']} final")
    console.print(table)

    console.print(f"\n[bold]Answer:[/] {result['answer']}\n")
    if result["retrieved_chunks"]:
        console.print("[bold]Chunks used:[/]")
        for i, c in enumerate(result["retrieved_chunks"], 1):
            score = c.get("rerank_score", c.get("rrf_score", c.get("similarity", 0)))
            tag = " [dim][expanded][/]" if c.get("is_expanded") else ""
            console.print(f"  [{i}] [cyan]{c['doc_name']}[/] chunk {c['chunk_index']}{tag} — {score:.4f}")

    if args.cache:
        console.print()
        console.print("[bold]Cache Stats:[/]", get_cache().stats())
