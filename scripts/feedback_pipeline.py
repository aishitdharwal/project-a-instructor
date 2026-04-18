"""
Feedback Pipeline — Session 7 (Project A, Instructor Version)

Closes the RAG improvement loop:

  1. run_pipeline_on_dataset()  — run golden dataset through ask(mode="advanced")
  2. run_ragas_eval()           — RAGAS faithfulness / answer_relevance /
                                  context_precision / context_recall
  3. find_weak_queries()        — bottom quartile or below 0.6 threshold
  4. compare_to_baseline()      — diff against baseline_scores.json
  5. correlate_with_rds_feedback() — join RAGAS scores with real user ratings
                                     from the RDS feedback table (Session 8)
  6. --generate flag            — generate synthetic candidates for docs linked
                                  to weak queries (via synthetic_generator)

Design decisions:
  - RAGAS wraps our retrieved chunks + generated answers into its Dataset format
  - We use langchain_openai.ChatOpenAI / OpenAIEmbeddings as RAGAS LLM wrappers
    (RAGAS judges need a LangChain-compatible LLM)
  - Weak query threshold: any RAGAS metric < 0.6 OR bottom quartile by score
  - RDS correlation: joins on exact query text; shows where human and automated
    scores agree and where they diverge (high RAGAS + thumbs down = blind spot)
  - Candidate queries saved to candidate_queries.json for human review
  - Results saved to feedback_results.json

Run:
  python -m scripts.feedback_pipeline
  python -m scripts.feedback_pipeline --generate
  python -m scripts.feedback_pipeline --save-baseline
  python -m scripts.feedback_pipeline --category membership
"""
import os
import sys
import json
import argparse
import statistics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

load_dotenv()

from scripts.rag import ask
from scripts.synthetic_generator import generate_questions

console = Console()

SCRIPT_DIR = os.path.dirname(__file__)
CORPUS_DIR = os.path.join(SCRIPT_DIR, "..", "corpus")
GOLDEN_DATASET_PATH = os.path.join(SCRIPT_DIR, "golden_dataset.json")
BASELINE_PATH = os.path.join(SCRIPT_DIR, "baseline_scores.json")
RESULTS_PATH = os.path.join(SCRIPT_DIR, "..", "feedback_results.json")
CANDIDATES_PATH = os.path.join(SCRIPT_DIR, "candidate_queries.json")

WEAK_THRESHOLD = 0.6


# =============================================================================
# STEP 1 — Run the pipeline on the golden dataset
# =============================================================================

def run_pipeline_on_dataset(
    queries: list[dict],
    mode: str = "advanced",
    guardrails: bool = False,
) -> list[dict]:
    """
    Run each golden dataset query through ask() and collect:
      - generated answer
      - retrieved context string
      - list of retrieved chunks (for source-level analysis)
      - trace_id for LangFuse linking
    """
    results = []
    for i, q in enumerate(queries):
        console.print(f"  [{i+1}/{len(queries)}] {q['query'][:70]}...", style="dim")
        try:
            r = ask(q["query"], mode=mode, guardrails=guardrails)
            results.append({
                "id": q["id"],
                "query": q["query"],
                "expected_answer": q.get("expected_answer", ""),
                "expected_source": q.get("expected_source", "N/A"),
                "difficulty": q.get("difficulty", "easy"),
                "category": q.get("category", "general"),
                "answer": r["answer"],
                "context": r["context"],
                "retrieved_chunks": r.get("retrieved_chunks", []),
                "trace_id": r.get("trace_id"),
                "cache_hit": r.get("cache_hit", False),
                "model_used": r.get("model_used", ""),
            })
        except Exception as e:
            console.print(f"  [red]Error on query {q['id']}: {e}[/]")
            results.append({
                "id": q["id"],
                "query": q["query"],
                "expected_answer": q.get("expected_answer", ""),
                "expected_source": q.get("expected_source", "N/A"),
                "difficulty": q.get("difficulty", "easy"),
                "category": q.get("category", "general"),
                "answer": "",
                "context": "",
                "retrieved_chunks": [],
                "trace_id": None,
                "cache_hit": False,
                "model_used": "",
                "error": str(e),
            })
    return results


# =============================================================================
# STEP 2 — RAGAS evaluation
# =============================================================================

def run_ragas_eval(pipeline_results: list[dict]) -> list[dict]:
    """
    Run RAGAS over the pipeline results.

    RAGAS metrics:
      - faithfulness:        Is the answer grounded in the retrieved context?
      - answer_relevancy:    Does the answer address the question?
      - context_precision:   Are the retrieved chunks relevant (signal-to-noise)?
      - context_recall:      Does the context contain what's needed for the answer?

    Each result dict gets a "ragas_scores" key with per-metric floats.
    """
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from datasets import Dataset

    # Build RAGAS-compatible dataset
    ragas_rows = []
    for r in pipeline_results:
        if not r.get("answer") or not r.get("context"):
            continue
        # RAGAS expects contexts as a list of strings
        chunks = r.get("retrieved_chunks", [])
        if chunks:
            contexts = [c["content"] for c in chunks if c.get("content")]
        else:
            # Fall back: split assembled context on separator
            contexts = [seg.strip() for seg in r["context"].split("---") if seg.strip()]

        ragas_rows.append({
            "question": r["query"],
            "answer": r["answer"],
            "contexts": contexts,
            "ground_truth": r["expected_answer"],
            "_id": r["id"],
        })

    if not ragas_rows:
        console.print("[red]No valid rows for RAGAS evaluation.[/]")
        return pipeline_results

    ds = Dataset.from_list(ragas_rows)

    # Wire RAGAS to use OpenAI
    llm_wrapper = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
    emb_wrapper = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

    for metric in [faithfulness, answer_relevancy, context_precision, context_recall]:
        metric.llm = llm_wrapper
    for metric in [answer_relevancy, context_precision, context_recall]:
        metric.embeddings = emb_wrapper

    console.print("\n[dim]Running RAGAS evaluation (faithfulness / answer_relevancy / context_precision / context_recall)...[/]")

    ragas_result = evaluate(
        dataset=ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    ragas_df = ragas_result.to_pandas()

    # Merge scores back into pipeline_results by position (RAGAS preserves order)
    id_to_scores = {}
    for i, row in ragas_df.iterrows():
        row_id = ragas_rows[i]["_id"]
        id_to_scores[row_id] = {
            "faithfulness":       round(float(row.get("faithfulness", 0) or 0), 4),
            "answer_relevancy":   round(float(row.get("answer_relevancy", 0) or 0), 4),
            "context_precision":  round(float(row.get("context_precision", 0) or 0), 4),
            "context_recall":     round(float(row.get("context_recall", 0) or 0), 4),
        }

    for r in pipeline_results:
        r["ragas_scores"] = id_to_scores.get(r["id"], {
            "faithfulness": None, "answer_relevancy": None,
            "context_precision": None, "context_recall": None,
        })

    return pipeline_results


# =============================================================================
# STEP 3 — Find weak queries
# =============================================================================

def find_weak_queries(
    results: list[dict],
    threshold: float = WEAK_THRESHOLD,
) -> list[dict]:
    """
    A query is "weak" if ANY RAGAS metric is below threshold, OR if it falls
    in the bottom quartile of the overall composite score.
    """
    scored = [r for r in results if r.get("ragas_scores") and
              any(v is not None for v in r["ragas_scores"].values())]

    def composite(r: dict) -> float:
        scores = [v for v in r["ragas_scores"].values() if v is not None]
        return sum(scores) / len(scores) if scores else 0.0

    composites = [composite(r) for r in scored]
    if len(composites) < 4:
        q1_threshold = threshold
    else:
        q1_threshold = sorted(composites)[len(composites) // 4]

    weak = []
    for r in scored:
        scores = r["ragas_scores"]
        below_threshold = any(
            v is not None and v < threshold
            for v in scores.values()
        )
        below_q1 = composite(r) <= q1_threshold
        if below_threshold or below_q1:
            r["composite_score"] = round(composite(r), 4)
            r["weak_reasons"] = [
                f"{k}<{threshold}" for k, v in scores.items()
                if v is not None and v < threshold
            ]
            weak.append(r)

    return weak


# =============================================================================
# STEP 4 — Compare to baseline
# =============================================================================

def compare_to_baseline(results: list[dict]) -> dict:
    """
    Compare current RAGAS averages against baseline_scores.json.
    Returns a dict with metric deltas.
    """
    if not os.path.exists(BASELINE_PATH):
        console.print("[dim]No baseline found — skipping comparison.[/]")
        return {}

    with open(BASELINE_PATH) as f:
        baseline = json.load(f)

    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    current = {}
    for metric in metrics:
        vals = [
            r["ragas_scores"][metric]
            for r in results
            if r.get("ragas_scores") and r["ragas_scores"].get(metric) is not None
        ]
        current[metric] = round(statistics.mean(vals), 4) if vals else None

    deltas = {}
    for metric in metrics:
        baseline_val = baseline.get("ragas", {}).get(metric)
        current_val = current.get(metric)
        if baseline_val is not None and current_val is not None:
            deltas[metric] = {
                "baseline": baseline_val,
                "current": current_val,
                "delta": round(current_val - baseline_val, 4),
            }

    return {"current": current, "deltas": deltas}


# =============================================================================
# STEP 5 — Generate synthetic candidates for weak doc sources
# =============================================================================

def generate_candidates_for_weak(weak_queries: list[dict], count: int = 3) -> list[dict]:
    """
    For each doc source that contributed to weak queries, generate new synthetic
    evaluation questions via synthetic_generator.generate_questions().

    Returns a flat list of candidate questions for human review.
    """
    # Collect unique doc sources from weak queries
    weak_sources = {
        r["expected_source"] for r in weak_queries
        if r.get("expected_source") and r["expected_source"] != "N/A"
    }

    candidates = []
    for doc_name in sorted(weak_sources):
        corpus_path = os.path.join(CORPUS_DIR, doc_name)
        if not os.path.exists(corpus_path):
            console.print(f"  [yellow]Corpus file not found: {doc_name} — skipping[/]")
            continue

        with open(corpus_path) as f:
            doc_text = f.read()

        console.print(f"  Generating candidates for [cyan]{doc_name}[/] (mismatch persona)...")
        new_qs = generate_questions(doc_name, doc_text, persona="mismatch", count=count)
        for q in new_qs:
            q["generated_for_weak"] = True
            q["doc_source"] = doc_name
        candidates.extend(new_qs)

    return candidates


# =============================================================================
# STEP 5 — Correlate RAGAS scores with real user feedback from RDS
# =============================================================================

def load_rds_feedback(source: str = "project-a") -> list[dict]:
    """
    Load user feedback rows from the RDS feedback table.
    Returns list of {query, rating, created_at} dicts.
    """
    import psycopg2
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
            """SELECT query, rating, created_at
               FROM feedback
               WHERE source = %s AND query IS NOT NULL
               ORDER BY created_at DESC""",
            (source,),
        )
        rows = [
            {"query": r[0], "rating": r[1], "created_at": str(r[2])}
            for r in cur.fetchall()
        ]
        cur.close()
        conn.close()
        return rows
    except Exception as e:
        console.print(f"[yellow]Could not load RDS feedback: {e}[/]")
        return []


def correlate_with_rds_feedback(results: list[dict], source: str = "project-a") -> list[dict]:
    """
    Join RAGAS pipeline results with user feedback from RDS on query text.

    Each matched result gets a "user_rating" key (+1 / -1) and an
    "agreement" key:
      "agree"    — RAGAS composite and user rating point the same direction
      "disagree" — high RAGAS but thumbs down, or low RAGAS but thumbs up
      "no_data"  — no user feedback found for this query
    """
    feedback_rows = load_rds_feedback(source)
    if not feedback_rows:
        console.print("[dim]No user feedback in RDS — skipping correlation.[/]")
        for r in results:
            r["user_rating"] = None
            r["agreement"] = "no_data"
        return results

    # Build lookup: query text → most recent rating
    feedback_map: dict[str, int] = {}
    for row in feedback_rows:
        if row["query"] and row["query"] not in feedback_map:
            feedback_map[row["query"]] = row["rating"]

    for r in results:
        rating = feedback_map.get(r["query"])
        r["user_rating"] = rating

        scores = r.get("ragas_scores") or {}
        vals = [v for v in scores.values() if v is not None]
        composite = sum(vals) / len(vals) if vals else None

        if rating is None or composite is None:
            r["agreement"] = "no_data"
        else:
            ragas_positive = composite >= WEAK_THRESHOLD
            user_positive = rating == 1
            r["agreement"] = "agree" if ragas_positive == user_positive else "disagree"

    return results


def display_feedback_correlation(results: list[dict]):
    """Show where RAGAS and user ratings agree/disagree."""
    matched = [r for r in results if r.get("user_rating") is not None]
    if not matched:
        console.print("[dim]No overlapping queries between RAGAS run and RDS feedback.[/]")
        return

    agree    = [r for r in matched if r["agreement"] == "agree"]
    disagree = [r for r in matched if r["agreement"] == "disagree"]

    console.print(Panel(
        f"[bold]Matched {len(matched)} queries with user feedback[/]\n"
        f"  [green]Agree:[/]    {len(agree)}  "
        f"  [red]Disagree:[/] {len(disagree)}",
        title="[bold yellow]RAGAS vs User Feedback Correlation[/]",
        border_style="yellow",
    ))

    if disagree:
        table = Table(title="Disagreements — investigate these", box=box.SIMPLE,
                      title_style="bold red")
        table.add_column("Query", width=48)
        table.add_column("RAGAS", justify="center", width=8)
        table.add_column("User", justify="center", width=6)
        table.add_column("Blind spot?", width=22)

        for r in disagree:
            scores = r.get("ragas_scores") or {}
            vals = [v for v in scores.values() if v is not None]
            composite = round(sum(vals) / len(vals), 3) if vals else None
            user = "👍" if r["user_rating"] == 1 else "👎"
            note = "High RAGAS, user unhappy" if r["user_rating"] == -1 else "Low RAGAS, user happy"
            table.add_row(
                r["query"][:46] + ".." if len(r["query"]) > 46 else r["query"],
                f"{composite:.3f}" if composite else "N/A",
                user,
                note,
            )
        console.print(table)


# =============================================================================
# DISPLAY HELPERS
# =============================================================================

def _score_color(val: float | None) -> str:
    if val is None:
        return "dim"
    if val >= 0.75:
        return "green"
    elif val >= 0.55:
        return "yellow"
    return "red"


def display_ragas_summary(results: list[dict]):
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    avgs = {}
    for m in metrics:
        vals = [r["ragas_scores"][m] for r in results
                if r.get("ragas_scores") and r["ragas_scores"].get(m) is not None]
        avgs[m] = round(statistics.mean(vals), 4) if vals else None

    table = Table(title="RAGAS Evaluation Summary", box=box.ROUNDED, title_style="bold cyan")
    table.add_column("Metric", style="bold", width=22)
    table.add_column("Score", justify="center", width=10)
    table.add_column("Interpretation", style="dim")

    interpretations = {
        "faithfulness":      "Answer claims grounded in retrieved context",
        "answer_relevancy":  "Answer directly addresses the question",
        "context_precision": "Retrieved chunks are relevant (low noise)",
        "context_recall":    "Context contains what's needed to answer",
    }
    for m in metrics:
        v = avgs[m]
        color = _score_color(v)
        score_str = f"[{color}]{v:.3f}[/]" if v is not None else "[dim]N/A[/]"
        table.add_row(m, score_str, interpretations[m])

    console.print(table)
    return avgs


def display_weak_queries(weak: list[dict]):
    if not weak:
        console.print("[green]No weak queries found — pipeline looks healthy.[/]")
        return

    table = Table(
        title=f"Weak Queries ({len(weak)} found)", box=box.SIMPLE, title_style="bold red"
    )
    table.add_column("ID", width=6)
    table.add_column("Query", width=46)
    table.add_column("Source", width=26)
    table.add_column("Composite", justify="center", width=9)
    table.add_column("Weak reasons", width=28)

    for r in sorted(weak, key=lambda x: x.get("composite_score", 0)):
        table.add_row(
            r["id"],
            r["query"][:44] + "..." if len(r["query"]) > 44 else r["query"],
            r.get("expected_source", "N/A")[:24],
            f"[{_score_color(r.get('composite_score'))}]{r.get('composite_score', 0):.3f}[/]",
            ", ".join(r.get("weak_reasons", [])) or "bottom quartile",
        )

    console.print(table)


def display_baseline_comparison(comparison: dict):
    if not comparison or not comparison.get("deltas"):
        return

    table = Table(
        title="vs Baseline", box=box.SIMPLE, title_style="bold yellow"
    )
    table.add_column("Metric", style="bold", width=22)
    table.add_column("Baseline", justify="center", width=10)
    table.add_column("Current", justify="center", width=10)
    table.add_column("Delta", justify="center", width=10)

    for metric, data in comparison["deltas"].items():
        delta = data["delta"]
        delta_color = "green" if delta >= 0 else "red"
        delta_sign = "+" if delta >= 0 else ""
        table.add_row(
            metric,
            f"{data['baseline']:.3f}",
            f"{data['current']:.3f}",
            f"[{delta_color}]{delta_sign}{delta:.3f}[/]",
        )

    console.print(table)


# =============================================================================
# SAVE BASELINE
# =============================================================================

def save_ragas_baseline(avgs: dict):
    existing = {}
    if os.path.exists(BASELINE_PATH):
        with open(BASELINE_PATH) as f:
            existing = json.load(f)

    existing["ragas"] = avgs

    with open(BASELINE_PATH, "w") as f:
        json.dump(existing, f, indent=2)

    console.print(Panel(
        f"[bold green]RAGAS baseline saved → {BASELINE_PATH}[/]\n"
        + "\n".join(f"  {k}: {v:.3f}" for k, v in avgs.items() if v is not None),
        title="[bold green]Baseline Locked[/]",
        border_style="green",
    ))


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Feedback pipeline: RAGAS eval + weak query detection")
    parser.add_argument("--mode", default="advanced",
                        choices=["dense", "hybrid", "advanced", "cached"],
                        help="RAG pipeline mode (default: advanced)")
    parser.add_argument("--guardrails", action="store_true",
                        help="Run guardrails on each query")
    parser.add_argument("--generate", action="store_true",
                        help="Generate synthetic candidates for weak doc sources")
    parser.add_argument("--save-baseline", action="store_true",
                        help="Save current RAGAS scores as new baseline")
    parser.add_argument("--category", type=str,
                        help="Filter golden dataset to a category prefix")
    parser.add_argument("--threshold", type=float, default=WEAK_THRESHOLD,
                        help=f"RAGAS score threshold for weak detection (default: {WEAK_THRESHOLD})")
    args = parser.parse_args()

    console.print(Panel(
        "[bold]Feedback Pipeline — Session 7[/]\n"
        f"[dim]Mode: {args.mode} | Guardrails: {args.guardrails} | "
        f"Generate candidates: {args.generate}[/]",
        title="[bold cyan]Project A — Feedback Loop[/]",
        border_style="cyan",
    ))

    # Load golden dataset
    if not os.path.exists(GOLDEN_DATASET_PATH):
        console.print(f"[red]golden_dataset.json not found at {GOLDEN_DATASET_PATH}[/]")
        console.print("[dim]Run scripts/synthetic_generator.py --merge to create it first.[/]")
        return

    with open(GOLDEN_DATASET_PATH) as f:
        golden = json.load(f)

    if args.category:
        golden = [q for q in golden if q.get("category", "").startswith(args.category)]
        console.print(f"[dim]Filtered to {len(golden)} queries in category: {args.category}[/]")

    console.print(f"\n[bold]Step 1:[/] Running pipeline on {len(golden)} queries...\n")
    results = run_pipeline_on_dataset(golden, mode=args.mode, guardrails=args.guardrails)

    console.print(f"\n[bold]Step 2:[/] RAGAS evaluation...\n")
    results = run_ragas_eval(results)

    console.print()
    avgs = display_ragas_summary(results)

    console.print(f"\n[bold]Step 3:[/] Finding weak queries (threshold={args.threshold})...\n")
    weak = find_weak_queries(results, threshold=args.threshold)
    display_weak_queries(weak)

    console.print(f"\n[bold]Step 4:[/] Comparing to baseline...\n")
    comparison = compare_to_baseline(results)
    display_baseline_comparison(comparison)

    console.print(f"\n[bold]Step 5:[/] Correlating with user feedback from RDS...\n")
    results = correlate_with_rds_feedback(results, source="project-a")
    display_feedback_correlation(results)

    if args.save_baseline:
        save_ragas_baseline(avgs)

    # Save full results
    output = {
        "summary": {
            "mode": args.mode,
            "total_queries": len(results),
            "weak_queries": len(weak),
            "ragas_averages": avgs,
        },
        "results": results,
        "weak_queries": weak,
        "baseline_comparison": comparison,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    console.print(f"\n[dim]Results saved → {RESULTS_PATH}[/]")

    # Optional: generate synthetic candidates for weak doc sources
    if args.generate and weak:
        console.print(f"\n[bold]Step 5:[/] Generating synthetic candidates for {len(weak)} weak queries...\n")
        candidates = generate_candidates_for_weak(weak)

        if candidates:
            with open(CANDIDATES_PATH, "w") as f:
                json.dump(candidates, f, indent=2, ensure_ascii=False)
            console.print(Panel(
                f"[bold green]{len(candidates)} candidate questions saved → {CANDIDATES_PATH}[/]\n"
                "[dim]Review and merge into golden_dataset.json with synthetic_generator.py --merge[/]",
                title="[bold green]Candidates Generated[/]",
                border_style="green",
            ))
        else:
            console.print("[yellow]No candidates generated (no matching corpus files).[/]")

    console.print()


if __name__ == "__main__":
    main()
