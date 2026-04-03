"""
Core RAG pipeline with LangFuse tracing.

Every stage is traced as a separate span:
  - query_embedding: embed the user query
  - retrieval: vector similarity search in pgvector
  - context_assembly: combine retrieved chunks into prompt context
  - generation: LLM call to produce the answer

This gives full visibility in LangFuse for the live debugging demo.

Usage:
    from scripts.rag import ask
    result = ask("What is the return policy for premium members?")
"""
import os
import json
import time
from openai import OpenAI
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
import psycopg2
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()
langfuse = Langfuse()

TOP_K = 5  # Number of chunks to retrieve

# Using gpt-4o-mini for generation — more realistic for production
# (most teams use cheaper models) and more sensitive to retrieval quality.
# GPT-4o can compensate for noisy context; mini can't. This makes
# retrieval failures visible in the output.
GENERATION_MODEL = "gpt-4o-mini"

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
def embed_query(query: str) -> list[float]:
    """Embed the user query."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query,
    )
    return response.data[0].embedding


@observe(name="retrieval")
def retrieve(query_embedding: list[float], top_k: int = TOP_K) -> list[dict]:
    """
    Vector similarity search in pgvector.
    Returns top_k most similar chunks with their similarity scores.
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT id, doc_name, chunk_index, content, metadata,
               1 - (embedding <=> %s::vector) AS similarity
        FROM chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (query_embedding, query_embedding, top_k),
    )

    results = []
    for row in cur.fetchall():
        results.append({
            "id": row[0],
            "doc_name": row[1],
            "chunk_index": row[2],
            "content": row[3],
            "metadata": row[4] if isinstance(row[4], dict) else json.loads(row[4]),
            "similarity": round(float(row[5]), 4),
        })

    cur.close()
    conn.close()

    # Log retrieval details to the current span
    langfuse_context.update_current_observation(
        metadata={
            "top_k": top_k,
            "results": [
                {
                    "doc_name": r["doc_name"],
                    "chunk_index": r["chunk_index"],
                    "similarity": r["similarity"],
                    "content_preview": r["content"][:100] + "...",
                }
                for r in results
            ],
        }
    )

    return results


@observe(name="context_assembly")
def assemble_context(retrieved_chunks: list[dict]) -> str:
    """
    Assemble retrieved chunks into a single context string.
    
    DELIBERATELY NAIVE: just concatenates chunks in retrieval order.
    No deduplication, no relevance filtering, no source attribution.
    """
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks):
        context_parts.append(
            f"[Source: {chunk['doc_name']}, Chunk {chunk['chunk_index']}, "
            f"Similarity: {chunk['similarity']}]\n{chunk['content']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    langfuse_context.update_current_observation(
        metadata={
            "num_chunks": len(retrieved_chunks),
            "total_context_chars": len(context),
            "sources": [c["doc_name"] for c in retrieved_chunks],
        }
    )

    return context


@observe(name="generation")
def generate(query: str, context: str) -> str:
    """Generate answer using GPT-4o-mini with the assembled context."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
        {"role": "user", "content": query},
    ]

    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=messages,
        temperature=0.1,
        max_tokens=1000,
    )

    answer = response.choices[0].message.content

    # Log generation details
    langfuse_context.update_current_observation(
        input=messages,
        output=answer,
        metadata={
            "model": GENERATION_MODEL,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
        usage={
            "input": response.usage.prompt_tokens,
            "output": response.usage.completion_tokens,
            "total": response.usage.total_tokens,
            "unit": "TOKENS",
        },
    )

    return answer


@observe(name="rag_pipeline")
def ask(query: str) -> dict:
    """
    Full RAG pipeline: embed → retrieve → assemble → generate.
    
    Returns a dict with:
      - query: the original question
      - answer: the generated response
      - retrieved_chunks: the raw retrieval results
      - context: the assembled context string
      - trace_id: LangFuse trace ID for debugging
    """
    start_time = time.time()

    # Tag the trace with query info
    langfuse_context.update_current_trace(
        input=query,
        metadata={"pipeline": "naive_rag", "top_k": TOP_K, "model": GENERATION_MODEL},
    )

    # Stage 1: Embed query
    query_embedding = embed_query(query)

    # Stage 2: Retrieve
    retrieved_chunks = retrieve(query_embedding)

    # Stage 3: Assemble context
    context = assemble_context(retrieved_chunks)

    # Stage 4: Generate
    answer = generate(query, context)

    elapsed = round(time.time() - start_time, 2)

    # Update trace with output
    langfuse_context.update_current_trace(
        output=answer,
        metadata={
            "elapsed_seconds": elapsed,
            "num_chunks_retrieved": len(retrieved_chunks),
        },
    )

    trace_id = langfuse_context.get_current_trace_id()

    langfuse.flush()

    return {
        "query": query,
        "answer": answer,
        "retrieved_chunks": retrieved_chunks,
        "context": context,
        "trace_id": trace_id,
        "elapsed_seconds": elapsed,
    }


if __name__ == "__main__":
    # Quick test
    result = ask("What is the standard return window for products?")
    print(f"\nQuery: {result['query']}")
    print(f"Answer: {result['answer']}")
    print(f"\nTrace ID: {result['trace_id']}")
    print(f"Time: {result['elapsed_seconds']}s")
    print(f"Model: {GENERATION_MODEL}")
    print(f"Chunks retrieved: {len(result['retrieved_chunks'])}")
    for i, chunk in enumerate(result["retrieved_chunks"]):
        print(f"  [{i+1}] {chunk['doc_name']} (chunk {chunk['chunk_index']}) — sim: {chunk['similarity']}")
    
    langfuse.flush()
