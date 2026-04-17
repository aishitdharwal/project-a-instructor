"""
Database Setup Script — Run once after deploying the base CloudFormation stack.

Creates the pgvector extension and the chunks table on Aurora PostgreSQL.
Safe to re-run (uses CREATE ... IF NOT EXISTS throughout).

What it creates:
  - vector extension (pgvector)
  - chunks table       — stores document chunks + embeddings
  - ivfflat index      — accelerates nearest-neighbour search
  - doc_name index     — accelerates metadata-filtered queries

Run locally:
    python -m scripts.setup_rds

Run via ECS one-off task (deploy.sh does this automatically):
    aws ecs run-task --overrides '{"containerOverrides":[{"command":["python","-m","scripts.setup_rds"]}]}'

Required environment variables (set by ECS task definition in production):
    PG_HOST, PG_PORT, PG_USER, PG_PASSWORD, PG_DATABASE
"""
import os
import sys
import psycopg2
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

load_dotenv()

console = Console()

# SQL statements executed in order
SETUP_SQL = [
    # 1. pgvector extension — must exist before VECTOR type can be used
    (
        "Enable pgvector extension",
        "CREATE EXTENSION IF NOT EXISTS vector;",
    ),
    # 2. Main chunks table — stores the knowledge base
    (
        "Create chunks table",
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id            SERIAL PRIMARY KEY,
            doc_name      TEXT        NOT NULL,
            chunk_index   INTEGER     NOT NULL,
            content       TEXT        NOT NULL,
            embedding     VECTOR(1536),           -- text-embedding-3-small dimension
            metadata      JSONB,
            created_at    TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE (doc_name, chunk_index)        -- prevents duplicate ingestion
        );
        """,
    ),
    # 3. IVFFlat index — approximate nearest-neighbour search (fast at query time)
    #    lists=100 is a good default for corpora up to ~100k chunks.
    #    Must be created AFTER data is loaded; here we create it empty (fine for setup).
    (
        "Create IVFFlat embedding index",
        """
        CREATE INDEX IF NOT EXISTS chunks_embedding_idx
        ON chunks USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
        """,
    ),
    # 4. B-tree index on doc_name — speeds up metadata-filtered retrieval
    (
        "Create doc_name index",
        "CREATE INDEX IF NOT EXISTS chunks_doc_name_idx ON chunks (doc_name);",
    ),
    # 5. Feedback table — user thumbs up/down, shared by project-a and project-b
    #    IF NOT EXISTS so re-running setup never wipes existing feedback rows.
    (
        "Create feedback table",
        """
        CREATE TABLE IF NOT EXISTS feedback (
            id         SERIAL PRIMARY KEY,
            trace_id   TEXT        NOT NULL,
            query      TEXT,
            rating     SMALLINT    NOT NULL,   -- +1 thumbs up, -1 thumbs down
            comment    TEXT,
            source     TEXT        DEFAULT 'project-a',
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """,
    ),
    # 6. Index on created_at — fast time-range queries on feedback
    (
        "Create feedback index",
        "CREATE INDEX IF NOT EXISTS feedback_created_at_idx ON feedback (created_at DESC);",
    ),
]


def get_connection():
    return psycopg2.connect(
        host=os.getenv("PG_HOST", "localhost"),
        port=int(os.getenv("PG_PORT", "5432")),
        user=os.getenv("PG_USER", "workshop"),
        password=os.getenv("PG_PASSWORD", "workshop123"),
        dbname=os.getenv("PG_DATABASE", "acmera_kb"),
        connect_timeout=10,
    )


def setup():
    console.print(Panel(
        f"[bold]Connecting to Aurora PostgreSQL[/]\n"
        f"[dim]Host: {os.getenv('PG_HOST', 'localhost')} | "
        f"DB: {os.getenv('PG_DATABASE', 'acmera_kb')}[/]",
        title="[bold cyan]Database Setup[/]",
        border_style="cyan",
    ))

    try:
        conn = get_connection()
        cur = conn.cursor()
    except Exception as e:
        console.print(f"[bold red]Connection failed:[/] {e}")
        sys.exit(1)

    console.print("[green]Connected.[/]\n")

    for label, sql in SETUP_SQL:
        console.print(f"  [dim]→[/] {label}...", end=" ")
        try:
            cur.execute(sql)
            conn.commit()
            console.print("[green]done[/]")
        except Exception as e:
            conn.rollback()
            console.print(f"[red]FAILED[/]\n    {e}")
            cur.close()
            conn.close()
            sys.exit(1)

    cur.close()
    conn.close()
    console.print("\n[bold green]Database setup complete.[/]")


if __name__ == "__main__":
    setup()
