"""
Step 2: Ingest documents into pgvector.

DELIBERATELY NAIVE chunking to create realistic production failures:
- Fixed-size character splitting (no semantic awareness)
- No overlap between chunks
- No document-type-specific handling
- Markdown headers get split from their content

This is how most tutorials teach RAG. It works on clean data.
It breaks on real documents.

Run: python scripts/ingest.py
"""
import os
import glob
import json
from openai import OpenAI
import psycopg2
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

CHUNK_SIZE = 500  # Characters — deliberately small to create splits
CHUNK_OVERLAP = 0  # No overlap — makes boundary problems worse
CORPUS_DIR = os.path.join(os.path.dirname(__file__), "..", "corpus")


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


def naive_chunk(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    """
    Fixed-size character splitting with no overlap.
    This is the 'tutorial-grade' chunking that breaks in production.
    
    Problems this creates:
    - Splits mid-sentence
    - Separates headers from their content
    - Breaks multi-part policies across chunks
    - No metadata about which section a chunk belongs to
    """
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i : i + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using OpenAI text-embedding-3-small."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [item.embedding for item in response.data]


def ingest():
    conn = get_connection()
    cur = conn.cursor()

    # Clear existing data
    cur.execute("DELETE FROM chunks;")

    doc_files = sorted(glob.glob(os.path.join(CORPUS_DIR, "*.md")))
    total_chunks = 0

    for filepath in doc_files:
        doc_name = os.path.basename(filepath)
        with open(filepath, "r") as f:
            content = f.read()

        chunks = naive_chunk(content)
        print(f"  {doc_name}: {len(chunks)} chunks")

        # Embed in batches of 20
        for batch_start in range(0, len(chunks), 20):
            batch = chunks[batch_start : batch_start + 20]
            embeddings = embed_texts(batch)

            for i, (chunk, embedding) in enumerate(zip(batch, embeddings)):
                chunk_index = batch_start + i
                metadata = json.dumps({
                    "doc_name": doc_name,
                    "chunk_index": chunk_index,
                    "char_start": chunk_index * CHUNK_SIZE,
                    "char_end": chunk_index * CHUNK_SIZE + len(chunk),
                })
                cur.execute(
                    """INSERT INTO chunks (doc_name, chunk_index, content, embedding, metadata)
                       VALUES (%s, %s, %s, %s::vector, %s)""",
                    (doc_name, chunk_index, chunk, embedding, metadata),
                )

        total_chunks += len(chunks)

    conn.commit()
    cur.close()
    conn.close()

    print(f"\nIngestion complete: {len(doc_files)} documents, {total_chunks} chunks.")
    print(f"Chunk size: {CHUNK_SIZE} chars, overlap: {CHUNK_OVERLAP} chars")
    print("WARNING: Using naive fixed-size chunking. This WILL create retrieval failures.")


if __name__ == "__main__":
    ingest()
